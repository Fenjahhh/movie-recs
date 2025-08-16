# api.py
# Endpunkt-Definitionen für das Movie-Recommender-System
from flask import Flask, request, jsonify
import json, numpy as np, pandas as pd
from pathlib import Path
import ast

app = Flask(__name__)

# --- Pfade & Lazy-Caches ---
DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")

MOVIES_FILE = (DATA_PROCESSED / "movies_processed.parquet"
               if (DATA_PROCESSED / "movies_processed.parquet").exists()
               else DATA_PROCESSED / "movies_processed.csv")

R_TRAIN_FILE = (DATA_PROCESSED / "ratings_train.parquet"
                if (DATA_PROCESSED / "ratings_train.parquet").exists()
                else DATA_PROCESSED / "ratings_train.csv")

_df_movies = None
_df_rtrain = None
_E = None
_meta = None

# --- Helpers ---

def to_py(obj):
    """Rekursiv NumPy/Pandas-Typen in JSON-fähige Python-Typen wandeln."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if obj is pd.NA:
        return None
    if isinstance(obj, np.ndarray):
        return [to_py(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    return obj  # str, bool, None, normale Python-Zahlen

def _ensure_loaded_content():
    """Nur Movies + Embeddings + Index laden (für /recommend/content)."""
    global _df_movies, _E, _meta
    if _df_movies is None:
        _df_movies = (pd.read_parquet(MOVIES_FILE)
                      if MOVIES_FILE.suffix == ".parquet"
                      else pd.read_csv(MOVIES_FILE)).reset_index(drop=True)
    if _E is None:
        if not (MODELS_DIR / "embeddings.npy").exists():
            raise FileNotFoundError("Embeddings fehlen. Bitte Tag 5 (--rebuild) ausführen.")
        _E = np.load(MODELS_DIR / "embeddings.npy")
    if _meta is None:
        with open(MODELS_DIR / "index.json", "r", encoding="utf-8") as f:
            _meta = json.load(f)

def _ensure_loaded_all():
    """Movies + Ratings-Train + Embeddings + Index laden (für personalisierte Route)."""
    global _df_movies, _df_rtrain, _E, _meta
    _ensure_loaded_content()
    if _df_rtrain is None:
        _df_rtrain = (pd.read_parquet(R_TRAIN_FILE)
                      if R_TRAIN_FILE.suffix == ".parquet"
                      else pd.read_csv(R_TRAIN_FILE))
        _df_rtrain["userId"] = _df_rtrain["userId"].astype(int)
        _df_rtrain["movieId"] = _df_rtrain["movieId"].astype(int)
        _df_rtrain["rating"]  = _df_rtrain["rating"].astype(float)

def _cosine_topk(E, vec, k=10, exclude=None):
    sims = E @ vec
    if exclude is not None and 0 <= exclude < sims.shape[0]:
        sims[exclude] = -1e9
    k = max(1, min(int(k), sims.shape[0]))  # bound k
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def _parse_genres(genres_val):
    """Genres robust in eine Python-Liste umwandeln (String/NDArray/Liste)."""
    if isinstance(genres_val, list):
        return genres_val
    if isinstance(genres_val, np.ndarray):
        return genres_val.tolist()
    if isinstance(genres_val, str):
        # Versuch, String-Liste sauber zu parsen
        try:
            parsed = ast.literal_eval(genres_val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # sonst Pipe-getrennt interpretieren
        return [g.strip() for g in genres_val.split("|") if g.strip()]
    return []

# --- Route: Content-Ähnlichkeit zu einem Titel/Text ---

@app.get("/recommend/content")
def recommend_content():
    try:
        _ensure_loaded_content()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    q = request.args.get("q", "").strip().lower()
    k_req = request.args.get("k", "10")
    try:
        k_req = int(k_req)
    except Exception:
        k_req = 10
    k = max(1, min(k_req, len(_df_movies) - 1))

    if not q:
        return jsonify({"error": "query param 'q' fehlt"}), 400

    # einfachen Titel-/Text-Match suchen
    mask = (
        _df_movies.get("clean_title", _df_movies["title"]).astype(str).str.lower().str.contains(q)
        | _df_movies["title"].astype(str).str.lower().str.contains(q)
        | _df_movies["content_text"].astype(str).str.lower().str.contains(q)
    )
    hits = _df_movies[mask]
    if hits.empty:
        return jsonify({"results": [], "note": "kein Treffer für q"}), 200

    r = int(hits.index[0])
    vec = _E[r]
    idxs, sims = _cosine_topk(_E, vec, k=k, exclude=r)

    results = []
    for i, s in zip(idxs, sims):
        row = _df_movies.iloc[i]
        genres = _parse_genres(row.get("genres_list", []))
        item = {
            "title": str(row["title"]),
            "movieId": int(row["movieId"]),
            "score": float(s),
            "genres": to_py(genres),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
        }
        results.append(item)

    return jsonify({
        "query": q,
        "anchor": str(_df_movies.loc[r, "title"]),
        "results": to_py(results)
    })

# --- Personalisierte Content-Empfehlungen ---

def _build_user_vec_api(user_id: int, mode: str, min_rating: float):
    # benötigt _ensure_loaded_all() vorher
    movieId_to_row = {int(k): int(v) for k, v in _meta["movieId_to_row"].items()}
    ur = _df_rtrain[_df_rtrain["userId"] == user_id]
    if ur.empty:
        return None, "user hat keine ratings"

    rows, weights = [], []
    for _, r in ur.iterrows():
        mid, rating = int(r["movieId"]), float(r["rating"])
        if mid not in movieId_to_row:
            continue
        row = movieId_to_row[mid]
        if mode == "pos_only":
            if rating >= min_rating:
                w = max(0.0, rating - min_rating)  # 0.5..1.5 bei min_rating 3.5
                rows.append(row); weights.append(w)
        else:  # mixed
            w = (rating - 3.0) / 2.0  # -1..+1
            w = float(np.clip(w, -1.0, 1.0))
            if abs(w) < 0.1:
                continue
            rows.append(row); weights.append(w)

    if not rows:
        return None, "zu wenig signal"

    V = _E[rows]; w = np.array(weights, dtype=np.float32)
    user_vec = (w[:, None] * V).sum(axis=0)
    n = np.linalg.norm(user_vec)
    if n < 1e-9:
        return None, "null vector"
    return (user_vec / n).astype(np.float32), set(rows)

@app.get("/recommend/content/personal")
def recommend_content_personal():
    try:
        _ensure_loaded_all()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # Parameter lesen
    try:
        user_id = int(request.args.get("userId", ""))
    except Exception:
        return jsonify({"error": "query param 'userId' (int) fehlt"}), 400
    k_req = request.args.get("k", "10")
    try:
        k_req = int(k_req)
    except Exception:
        k_req = 10
    k = max(1, min(k_req, len(_df_movies) - 1))
    mode = request.args.get("mode", "pos_only")
    min_rating = float(request.args.get("minRating", 3.5))

    user_vec, seen_rows = _build_user_vec_api(user_id, mode, min_rating)
    if user_vec is None:
        return jsonify({"userId": user_id, "results": [], "note": "kein Profil"}), 200

    sims = _E @ user_vec
    for r in seen_rows:
        if 0 <= r < sims.shape[0]:
            sims[r] = -1e9
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    scores = sims[idx]

    row_to_movieId = _meta["row_to_movieId"]
    results = []
    for i, s in zip(idx, scores):
        mid = int(row_to_movieId[i])
        m = _df_movies.iloc[i]  # gleiche Reihenfolge wie Embeddings
        results.append({
            "movieId": mid,
            "title": str(m["title"]),
            "year": int(m["year"]) if "year" in m and pd.notna(m["year"]) else None,
            "genres": to_py(_parse_genres(m.get("genres_list", []))),
            "score": float(s),
        })

    return jsonify({"userId": user_id, "mode": mode, "k": k, "results": to_py(results)})

# --- Healthcheck ---
@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
