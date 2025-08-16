# scripts/topn_cf_demo.py
import joblib, pandas as pd
from pathlib import Path
from surprise import Dataset, Reader

DATA = Path("data/processed")
MODELS = Path("models")

def load_surprise_trainset(df_train):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_train[["userId","movieId","rating"]], reader)
    return data.build_full_trainset()

def topn_for_user(uid: int, algo, trainset, movies_df: pd.DataFrame, N=10):
    # Gesehene Items des Users im Trainset
    inner_uid = trainset.to_inner_uid(uid)
    seen_inner = set(j for (j, _) in trainset.ur[inner_uid])

    # Kandidaten = alle Items, die der User noch nicht gesehen hat
    candidates_inner = [j for j in trainset.all_items() if j not in seen_inner]

    preds = []
    for j in candidates_inner:
        iid = trainset.to_raw_iid(j)   # movieId
        pred = algo.predict(uid, int(iid), verbose=False)
        preds.append((int(iid), pred.est))

    # Top-N nach est
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:N]

    # Anreichern mit Titel/Jahr
    m = movies_df.set_index("movieId")
    out = []
    for mid, est in top:
        row = m.loc[mid]
        out.append({"movieId": mid, "title": str(row["title"]),
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "est": float(est)})
    return out

if __name__ == "__main__":
    algo = joblib.load(MODELS / "cf_svd.pkl")
    df_train = pd.read_parquet(DATA / "ratings_train.parquet")
    movies   = pd.read_parquet(DATA / "movies_processed.parquet")
    trainset = load_surprise_trainset(df_train)

    res = topn_for_user(uid=1, algo=algo, trainset=trainset, movies_df=movies, N=10)
    for i, r in enumerate(res, 1):
        print(f"{i:2d}. {r['title']} ({r.get('year')})  est={r['est']:.2f}")
