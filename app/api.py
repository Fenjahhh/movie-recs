# api.py
# Endpunkt-Definitionen für das Movie-Recommender-System
from flask import Flask, request, jsonify
import json, numpy as np, pandas as pd
from pathlib import Path

app = Flask(__name__)


# --- einmalig laden (Lazy-Init geht auch) ---
DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
MOVIES_FILE = (DATA_PROCESSED / "movies_processed.parquet"
               if (DATA_PROCESSED / "movies_processed.parquet").exists()
               else DATA_PROCESSED / "movies_processed.csv")

_df_movies = None
_E = None
_meta = None

def _ensure_loaded():
    global _df_movies, _E, _meta
    if _df_movies is None:
        _df_movies = (pd.read_parquet(MOVIES_FILE)
                      if MOVIES_FILE.suffix == ".parquet"
                      else pd.read_csv(MOVIES_FILE))
        _df_movies = _df_movies.reset_index(drop=True)
    if _E is None:
        _E = np.load(MODELS_DIR / "embeddings.npy")
    if _meta is None:
        with open(MODELS_DIR / "index.json", "r", encoding="utf-8") as f:
            _meta = json.load(f)

def _cosine_topk(E, vec, k=10, exclude=None):
    sims = E @ vec
    if exclude is not None:
        sims[exclude] = -1e9
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

@app.get("/recommend/content")
def recommend_content():
    _ensure_loaded()
    q = request.args.get("q", "").strip().lower()
    k = int(request.args.get("k", 10))
    if not q:
        return jsonify({"error": "query param 'q' fehlt"}), 400

    # einfachen Titel-Match suchen
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
        results.append({
            "title": str(row["title"]),
            "movieId": int(row["movieId"]),
            "score": float(s),
            "genres": row.get("genres_list", []),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None
        })
    return jsonify({"query": q, "anchor": str(_df_movies.loc[r, "title"]), "results": results})

@app.get("/health")
def health():
    return {"ok": True}
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
