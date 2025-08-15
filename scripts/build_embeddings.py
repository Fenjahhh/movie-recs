# scripts/build_embeddings.py
# Erzeugt Embeddings aus data/processed/movies_processed.parquet
# Speichert: models/embeddings.npy + models/index.json
# Optional: Suche nach ähnlichen Filmen via --search "Titel"

from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MOVIES_PARQUET = DATA_PROCESSED / "movies_processed.parquet"
MOVIES_CSV_FALLBACK = DATA_PROCESSED / "movies_processed.csv"

INDEX_JSON = MODELS_DIR / "index.json"
EMB_NPY = MODELS_DIR / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # klein & gut

def load_movies():
    if MOVIES_PARQUET.exists():
        df = pd.read_parquet(MOVIES_PARQUET)
    elif MOVIES_CSV_FALLBACK.exists():
        df = pd.read_csv(MOVIES_CSV_FALLBACK)
    else:
        raise FileNotFoundError("movies_processed.* nicht gefunden. Bitte Tag 3 ausführen.")
    # Minimal-Checks
    assert "movieId" in df and "content_text" in df, "Spalten movieId/content_text fehlen."
    df = df.copy()
    df["movieId"] = df["movieId"].astype(int)
    df["content_text"] = df["content_text"].fillna(df.get("clean_title", "")).astype(str)
    return df

def build_and_save_embeddings(df, batch_size=128):
    texts = df["content_text"].tolist()
    print(f"Modell laden: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"Berechne Embeddings für {len(texts)} Filme ...")
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # wichtig für Cosine-Ähnlichkeit
    )
    embs = np.asarray(embs, dtype=np.float32)  # kompakter
    np.save(EMB_NPY, embs)
    print(f"Gespeichert: {EMB_NPY} mit Shape {embs.shape}")

    # Mapping sichern
    row_to_movieId = df["movieId"].astype(int).tolist()
    movieId_to_row = {int(mid): i for i, mid in enumerate(row_to_movieId)}
    meta = {
        "model_name": MODEL_NAME,
        "row_to_movieId": row_to_movieId,
        "movieId_to_row": movieId_to_row,
    }
    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Gespeichert: {INDEX_JSON}")

def load_index_and_embeddings():
    with open(INDEX_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)
    embs = np.load(EMB_NPY)
    return meta, embs

def cosine_topk(E: np.ndarray, vec: np.ndarray, k=10, exclude: int | None = None):
    # E und vec sind L2-normalisiert (durch normalize_embeddings=True)
    sims = E @ vec  # Cosine
    if exclude is not None:
        sims[exclude] = -1e9  # sich selbst ausschließen
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def find_row_by_title(df: pd.DataFrame, query: str):
    q = query.strip().lower()
    # Suche in clean_title, title und content_text (einfach)
    mask = (
        df.get("clean_title", df["content_text"]).str.lower().str.contains(re.escape(q), na=False)
        | df["title"].astype(str).str.lower().str.contains(re.escape(q), na=False)
        | df["content_text"].str.lower().str.contains(re.escape(q), na=False)
    )
    hits = df[mask]
    if hits.empty:
        return None
    # nimm den ersten Treffer
    return int(hits.index[0])

def search_similar(df, query: str, k=10):
    print(f"Suche ähnlich zu: {query!r}")
    meta, E = load_index_and_embeddings()
    row_to_movieId = meta["row_to_movieId"]
    movieId_to_row = meta["movieId_to_row"]

    # Falls Reihenfolge sich geändert hat: stelle sicher, dass df gleiche Reihenfolge hat
    # (Hier gehen wir davon aus: df entspricht der Reihenfolge beim Embedding-Bau.)
    r = find_row_by_title(df, query)
    if r is None:
        print("Kein Treffer für den Suchtext.")
        return

    vec = E[r]
    idxs, sims = cosine_topk(E, vec, k=k, exclude=r)
    print("\nTop-Ähnlich:")
    for rank, (i, s) in enumerate(zip(idxs, sims), start=1):
        mid = row_to_movieId[i]
        row = df.index[i]
        title = df.loc[row, "title"]
        print(f"{rank:2d}. {title}  (score={s:.3f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Embeddings neu erzeugen und speichern.")
    ap.add_argument("--search", type=str, default=None, help='Suche ähnliche Filme zu diesem Text/Titel.')
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    df = load_movies()
    df = df.reset_index(drop=True)  # feste Reihenfolge

    if args.rebuild or not (EMB_NPY.exists() and INDEX_JSON.exists()):
        build_and_save_embeddings(df)

    if args.search:
        search_similar(df, args.search, k=args.k)
    else:
        # kleiner Hinweis, wie man die Suche startet
        print("\nEmbeddings sind bereit. Beispielsuche:")
        print('  python scripts/build_embeddings.py --search "toy story" --k 10')

if __name__ == "__main__":
    main()
