# api.py
# Endpunkt-Definitionen für das Movie-Recommender-System
from flask import Flask, request, jsonify
import json, numpy as np, pandas as pd
from pathlib import Path
import ast
import joblib
from surprise import Dataset, Reader

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

_cf_algo = None
_trainset = None

def _ensure_cf_loaded():
    global _cf_algo, _trainset
    if _cf_algo is None:
        _cf_algo = joblib.load(MODELS_DIR / "cf_svd.pkl")
    if _trainset is None:
        # ratings_train laden und Trainset für Surprise bauen
        rfile = (DATA_PROCESSED / "ratings_train.parquet"
                 if (DATA_PROCESSED / "ratings_train.parquet").exists()
                 else DATA_PROCESSED / "ratings_train.csv")
        df_train = (pd.read_parquet(rfile) if rfile.suffix==".parquet" else pd.read_csv(rfile))
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df_train[["userId","movieId","rating"]], reader)
        _trainset = data.build_full_trainset()

def _popular_topn(movies_df: pd.DataFrame, ratings_train_df: pd.DataFrame, N: int = 10, min_count: int = 20):
    """
    Beliebte Filme nach gewichteter Bewertung:
    WR = (v/(v+m))*R + (m/(v+m))*C
    v=count(movie), R=mean(movie), C=global mean, m=Schwelle (min_count)
    """
    grp = ratings_train_df.groupby("movieId")["rating"].agg(["mean", "count"]).rename(columns={"mean":"R","count":"v"})
    if len(grp) == 0:
        return []

    C = grp["R"].mean()
    m = min_count
    grp["WR"] = (grp["v"]/(grp["v"]+m))*grp["R"] + (m/(grp["v"]+m))*C
    top = grp.sort_values(["WR","v"], ascending=False).head(N)

    mdf = movies_df.set_index("movieId")
    out = []
    for mid, row in top.iterrows():
        if mid not in mdf.index:
            continue
        mrow = mdf.loc[mid]
        out.append({
            "movieId": int(mid),
            "title": str(mrow["title"]),
            "year": int(mrow["year"]) if "year" in mrow and pd.notna(mrow["year"]) else None,
            "est": float(row["WR"]),
            "count": int(row["v"]),
        })
    return out

def _cf_estimates_for_items(uid: int, item_ids: list[int]) -> dict[int, float]:
    """
    Liefert CF-Schätzungen (est) für eine Liste von movieIds.
    Gibt nur Items zurück, die im CF-Trainset bekannt sind.
    """
    ests = {}
    # Ist der User im Trainset vorhanden?
    try:
        _trainset.to_inner_uid(uid)
    except ValueError:
        return ests  # leer -> caller kann Hybrid ohne CF rechnen

    for mid in item_ids:
        try:
            # Prüfen, ob Item im Trainset existiert
            _trainset.to_inner_iid(str(mid))
        except ValueError:
            continue  # unbekanntes Item im CF-Trainset
        est = _cf_algo.predict(uid, int(mid), verbose=False).est
        ests[int(mid)] = float(est)
    return ests


def _cf_topn(uid: int, N: int, movies_df: pd.DataFrame):
    # Sicherstellen, dass User/Items im Trainset existieren
    try:
        inner_uid = _trainset.to_inner_uid(uid)
    except ValueError:
        return []  # User unbekannt im Train → keine CF-Empfehlungen

    seen_inner = set(j for (j, _) in _trainset.ur[inner_uid])
    candidates_inner = [j for j in _trainset.all_items() if j not in seen_inner]

    preds = []
    for j in candidates_inner:
        iid = int(_trainset.to_raw_iid(j))
        est = _cf_algo.predict(uid, iid, verbose=False).est
        preds.append((iid, est))

    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:N]
    m = movies_df.set_index("movieId")

    results = []
    for mid, est in top:
        if mid not in m.index:
            continue
        row = m.loc[mid]
        results.append({
            "movieId": int(mid),
            "title": str(row["title"]),
            "year": int(row["year"]) if "year" in row and pd.notna(row["year"]) else None,
            "est": float(est)
        })
    return results
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

@app.get("/recommend/cf/personal")
def recommend_cf_personal():
    try:
        _ensure_loaded_content()  # Movies + Embeddings/Index (Movies brauchen wir hier)
        _ensure_cf_loaded()       # CF-Modell + Surprise-Trainset
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # Parameter
    try:
        uid = int(request.args.get("userId", ""))
    except Exception:
        return jsonify({"error": "query param 'userId' (int) fehlt"}), 400

    try:
        k_req = int(request.args.get("k", 10))
    except Exception:
        k_req = 10
    k = max(1, min(k_req, 50))

    # Wenn User im Trainset bekannt → CF
    try:
        _trainset.to_inner_uid(uid)
        cf_results = _cf_topn(uid, k, _df_movies)
        if cf_results:
            return jsonify({"userId": uid, "k": k, "method": "cf", "results": to_py(cf_results)})
    except ValueError:
        pass  # user unbekannt → Fallback

    # Fallback: populäre/gute Filme (Cold-Start)
    # ratings_train laden (wir haben schon _trainset, aber für die Aggregation brauchen wir DataFrame):
    rfile = (DATA_PROCESSED / "ratings_train.parquet"
             if (DATA_PROCESSED / "ratings_train.parquet").exists()
             else DATA_PROCESSED / "ratings_train.csv")
    df_train = (pd.read_parquet(rfile) if rfile.suffix == ".parquet" else pd.read_csv(rfile))
    df_train["userId"] = df_train["userId"].astype(int)
    df_train["movieId"] = df_train["movieId"].astype(int)
    df_train["rating"]  = df_train["rating"].astype(float)

    pop_results = _popular_topn(_df_movies, df_train, N=k, min_count=20)
    return jsonify({"userId": uid, "k": k, "method": "popular_fallback", "results": to_py(pop_results)})

@app.get("/recommend/hybrid/personal")
def recommend_hybrid_personal():
    """
    Kombiniert Content-Score (Cosine zu User-Embedding) und CF-Score (SVD est).
    Params:
      userId (int, required)
      k (int, default=10)
      alpha (float in [0,1], default=0.6)  -> Gewicht für CF
      mode ("pos_only"|"mixed", default="pos_only")
      minRating (float, default=3.5)
      cand (int, default=200)  -> Größe des Kandidatenpools aus Content-Top-K
    Fallbacks:
      - hat User kein Profil -> Popular-Fallback
      - keine CF-Schätzungen verfügbar -> nur Content
    """
    # Laden
    try:
        _ensure_loaded_all()  # Movies + Ratings + Embeddings + Index
        _ensure_cf_loaded()   # CF-Modell + Surprise-Trainset
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # Parameter
    try:
        uid = int(request.args.get("userId", ""))
    except Exception:
        return jsonify({"error": "query param 'userId' (int) fehlt"}), 400

    def _to_int(x, default):
        try: return int(x)
        except: return default

    def _to_float(x, default):
        try:
            v = float(x); 
            return v
        except:
            return default

    k_req   = _to_int(request.args.get("k", 10), 10)
    k       = max(1, min(k_req, 50))
    alpha   = _to_float(request.args.get("alpha", 0.6), 0.6)
    alpha   = min(1.0, max(0.0, alpha))
    mode    = request.args.get("mode", "pos_only")
    min_r   = _to_float(request.args.get("minRating", 3.5), 3.5)
    cand    = _to_int(request.args.get("cand", 200), 200)
    cand    = max(k, min(cand, len(_df_movies)))

    # User-Embedding holen
    user_vec_res = _build_user_vec_api(uid, mode, min_r)
    if user_vec_res[0] is None:
        # Kein Content-Profil -> Popular-Fallback (Cold Start)
        rfile = (DATA_PROCESSED / "ratings_train.parquet"
                 if (DATA_PROCESSED / "ratings_train.parquet").exists()
                 else DATA_PROCESSED / "ratings_train.csv")
        df_train = (pd.read_parquet(rfile) if rfile.suffix == ".parquet" else pd.read_csv(rfile))
        df_train["userId"] = df_train["userId"].astype(int)
        df_train["movieId"] = df_train["movieId"].astype(int)
        df_train["rating"]  = df_train["rating"].astype(float)
        pop = _popular_topn(_df_movies, df_train, N=k, min_count=20)
        return jsonify({"userId": uid, "k": k, "method": "popular_fallback", "results": to_py(pop)})

    user_vec, seen_rows = user_vec_res

    # Content-Scores (Cosine) über alle Items, gesehene ausschließen
    sims = _E @ user_vec  # Cosine in [-1,1]
    for r in seen_rows:
        if 0 <= r < sims.shape[0]:
            sims[r] = -1e9

    # Kandidaten: Top-cand nach Content
    cand = max(k, min(cand, sims.shape[0]-len(seen_rows)))
    idx = np.argpartition(-sims, cand-1)[:cand]
    idx = idx[np.argsort(-sims[idx])]
    sims_cand = sims[idx]           # Cosine (roh)
    mids_cand = [int(_meta["row_to_movieId"][int(i)]) for i in idx]

    # Normalisiere Content: [-1,1] -> [0,1]
    content_norm = ((sims_cand + 1.0) / 2.0).astype(float)

    # CF-Schätzungen für Kandidaten
    cf_est = _cf_estimates_for_items(uid, mids_cand)
    # Normiere CF: [0.5,5.0] -> [0,1]
    def cf_to_norm(e): return max(0.0, min(1.0, (e - 0.5) / 4.5))

    # Hybrid bilden
    scores = []
    for rank, (i_row, mid, c_s) in enumerate(zip(idx, mids_cand, content_norm)):
        cf_s = cf_est.get(mid, None)
        if cf_s is None:
            h = (1.0 - alpha) * c_s  # kein CF -> nur Content-Anteil
        else:
            h = alpha * cf_to_norm(cf_s) + (1.0 - alpha) * c_s
        scores.append((h, c_s, cf_s, int(i_row), int(mid)))

    # sortiere nach Hybrid desc, nimm Top-k
    scores.sort(key=lambda t: t[0], reverse=True)
    top = scores[:k]

    # Ausgaben zusammenbauen
    results = []
    for h, c_s, cf_s, row_i, mid in top:
        m = _df_movies.iloc[row_i]
        results.append({
            "movieId": int(mid),
            "title": str(m["title"]),
            "year": int(m["year"]) if "year" in m and pd.notna(m["year"]) else None,
            "genres": to_py(_parse_genres(m.get("genres_list", []))),
            "score": {
                "hybrid": float(h),
                "content_norm": float(c_s),
                "cf_est": float(cf_s) if cf_s is not None else None
            }
        })

    return jsonify({
        "userId": uid,
        "k": k,
        "alpha": alpha,
        "method": "hybrid",
        "candidates": int(len(scores)),
        "results": to_py(results)
    })


# --- Healthcheck ---
@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
