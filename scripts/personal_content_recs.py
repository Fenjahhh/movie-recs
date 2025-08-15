# scripts/personal_content_recs.py
# Persönliche Empfehlungen aus:
# - data/processed/ratings_train.parquet (oder CSV-Fallback)
# - data/processed/movies_processed.parquet (oder CSV-Fallback)
# - models/embeddings.npy + models/index.json (von Tag 5)
#
# Nutzung:
#   python scripts/personal_content_recs.py --user-id 1 --k 10
#   python scripts/personal_content_recs.py --list-top-users 20
#
# API-Logik bauen wir im Anschluss in app/api.py ein.

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

DATA = Path("data/processed")
MODELS = Path("models")

# ---- Laden der Datengrundlage ----

def _read_parquet_or_csv(path_parquet: Path, csv_name: str):
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
    csv_path = path_parquet.with_name(csv_name)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Weder {path_parquet} noch {csv_path} gefunden.")

def load_movies():
    # movies_processed.* kommt aus Tag 3
    p = DATA / "movies_processed.parquet"
    df = _read_parquet_or_csv(p, "movies_processed.csv")
    df = df.reset_index(drop=True)
    # Minimal: movieId muss int sein
    df["movieId"] = df["movieId"].astype(int)
    return df

def load_ratings_train():
    # aus Tag 4
    p = DATA / "ratings_train.parquet"
    df = _read_parquet_or_csv(p, "ratings_train.csv")
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df

def load_index_and_embeddings():
    with open(MODELS / "index.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    E = np.load(MODELS / "embeddings.npy")
    # Sicherheit: Embeddings sind bereits normalisiert (Tag 5), sonst hier normalisieren:
    # E = E / np.maximum(1e-9, np.linalg.norm(E, axis=1, keepdims=True))
    return meta, E

# ---- Core: Nutzervektor + Empfehlung ----

def build_user_vector(user_ratings: pd.DataFrame,
                      movieId_to_row: dict[int, int],
                      E: np.ndarray,
                      mode: str = "pos_only",
                      min_rating: float = 3.5):
    """
    Erzeuge einen Nutzervektor aus seinen Bewertungen.
    mode:
      - 'pos_only': verwende nur positive Bewertungen (>= min_rating), Gewicht = (rating - min_rating)
      - 'mixed':    positive ziehen an, negative schieben weg:
                    Gewicht = (rating - 3.0) / 2.0  -> 1★=-1.0 ... 5★=+1.0 (geclippt)
    """
    rows = []
    weights = []

    for _, r in user_ratings.iterrows():
        mid = int(r["movieId"])
        rating = float(r["rating"])
        if mid not in movieId_to_row:
            continue  # Film ohne Embedding (sollte selten sein)
        row = movieId_to_row[mid]

        if mode == "pos_only":
            if rating >= min_rating:
                w = max(0.0, rating - min_rating)  # 0.5..1.5 bei min_rating=3.5
                rows.append(row)
                weights.append(w)
        else:  # mixed
            w = (rating - 3.0) / 2.0  # -1..+1
            w = float(np.clip(w, -1.0, 1.0))
            # Optional: ganz schwache |w| verwerfen
            if abs(w) < 0.1:
                continue
            rows.append(row)
            weights.append(w)

    if not rows:
        return None  # zu wenig Signal

    V = E[rows]                                # (n, d)
    w = np.array(weights, dtype=np.float32)    # (n,)
    # gewichteter Mittelwert
    user_vec = (w[:, None] * V).sum(axis=0)
    # normalisieren (für Cosine)
    norm = np.linalg.norm(user_vec)
    if norm < 1e-9:
        return None
    user_vec = (user_vec / norm).astype(np.float32)
    return user_vec, set(rows)  # rows merken, um gesehene Filme auszuschließen

def recommend_for_user(user_id: int,
                       ratings_train: pd.DataFrame,
                       movies: pd.DataFrame,
                       meta: dict,
                       E: np.ndarray,
                       k: int = 10,
                       mode: str = "pos_only",
                       min_rating: float = 3.5):
    # Ratings dieses Users
    ur = ratings_train[ratings_train["userId"] == user_id]
    if ur.empty:
        raise ValueError(f"User {user_id} hat keine Trainingsratings.")

    movieId_to_row = {int(k): int(v) for k, v in meta["movieId_to_row"].items()}
    res = build_user_vector(ur, movieId_to_row, E, mode=mode, min_rating=min_rating)
    if res is None:
        return {"userId": user_id, "results": [], "note": "zu wenig Signal in den Ratings"}
    user_vec, seen_rows = res

    # Cosine via Dot-Product (E ist normalisiert, user_vec normalisiert)
    sims = E @ user_vec  # (n,)
    # gesehene Filme rauswerfen
    for r in seen_rows:
        sims[r] = -1e9

    # Top-K holen
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    scores = sims[idx]

    # Index → movieId → Info
    row_to_movieId = [int(x) for x in meta["row_to_movieId"]]
    out = []
    for i, s in zip(idx, scores):
        mid = row_to_movieId[i]
        # movies ist in derselben Reihenfolge wie Embeddings entstanden (Tag 5 reset_index)
        # Falls Reihenfolge unsicher ist, lieber über movieId joinen:
        row = movies.index[i]
        title = str(movies.loc[row, "title"])
        year = movies.loc[row, "year"] if "year" in movies.columns else None
        genres = movies.loc[row, "genres_list"] if "genres_list" in movies.columns else []
        out.append({
            "movieId": int(mid),
            "title": title,
            "year": int(year) if pd.notna(year) else None,
            "genres": genres if isinstance(genres, list) else [],
            "score": float(s)
        })

    return {"userId": user_id, "mode": mode, "k": k, "results": out}

# ---- kleine Helfer ----

def list_top_users(ratings_train: pd.DataFrame, n=20):
    c = ratings_train.groupby("userId")["movieId"].count().sort_values(ascending=False)
    return c.head(n)

# ---- CLI ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, default=None, help="UserID aus ratings_train wählen")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", type=str, default="pos_only", choices=["pos_only", "mixed"])
    ap.add_argument("--min-rating", type=float, default=3.5, help="Schwelle für pos_only")
    ap.add_argument("--list-top-users", type=int, default=0, help="Zeige die aktivsten N Nutzer")
    args = ap.parse_args()

    movies = load_movies()
    ratings_train = load_ratings_train()
    meta, E = load_index_and_embeddings()

    if args.list_top_users:
        print(list_top_users(ratings_train, args.list_top_users))
        return

    if args.user_id is None:
        # pragmatisch: nimm einfach einen aktiven Nutzer
        uid = int(list_top_users(ratings_train, 1).index[0])
        print(f"Keine --user-id angegeben. Nehme aktiven Nutzer {uid}.")
        args.user_id = uid

    recs = recommend_for_user(args.user_id, ratings_train, movies, meta, E,
                              k=args.k, mode=args.mode, min_rating=args.min_rating)
    print(f"\nTop-{args.k} Empfehlungen für userId={args.user_id} (mode={args.mode}):")
    for i, r in enumerate(recs["results"], start=1):
        y = f" ({r['year']})" if r.get("year") else ""
        print(f"{i:2d}. {r['title']}{y}  score={r['score']:.3f}")

if __name__ == "__main__":
    main()
