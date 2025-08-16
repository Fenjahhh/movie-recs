# scripts/train_cf.py
# Trainiert eine Surprise-SVD-CF-Baseline auf ratings_train und evaluiert auf val/test.
# Speichert Modell + Meta in models/.

from pathlib import Path
import argparse, json, joblib
import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV

DATA = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

def read_parquet_or_csv(base: Path, name: str):
    pq = base / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = base / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Weder {pq} noch {csv} gefunden.")

def to_surprise_data(df: pd.DataFrame, rating_scale=(0.5, 5.0)):
    df = df[["userId", "movieId", "rating"]].copy()
    # Surprise schluckt ints/strings – wir lassen ints.
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df, reader)
    return data

def build_trainset(df_train: pd.DataFrame):
    data = to_surprise_data(df_train)
    return data.build_full_trainset()

def filter_to_known(trainset, df_eval: pd.DataFrame):
    """Nur Paare (userId,movieId) behalten, die im Trainset vorkommen."""
    known_u = set(trainset._raw2inner_id_users.keys())
    known_i = set(trainset._raw2inner_id_items.keys())
    rows = []
    for u, i, r in df_eval[["userId","movieId","rating"]].itertuples(index=False):
        if (u in known_u) and (i in known_i):
            rows.append((u, i, float(r)))
    return rows

def evaluate(algo, trainset, df_eval: pd.DataFrame, name: str):
    testset = filter_to_known(trainset, df_eval)
    if len(testset) == 0:
        print(f"[WARN] {name}: keine auswertbaren Paare (alle unbekannt im Trainset).")
        return {"name": name, "n": 0, "rmse": None, "mae": None}
    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae  = accuracy.mae(preds,  verbose=False)
    print(f"[{name}] n={len(testset)}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"name": name, "n": len(testset), "rmse": rmse, "mae": mae}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="kleine Grid-Suche auf Train (3-fold CV)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Lade Splits ...")
    df_train = read_parquet_or_csv(DATA, "ratings_train")
    df_val   = read_parquet_or_csv(DATA, "ratings_val")
    df_test  = read_parquet_or_csv(DATA, "ratings_test")

    # Datentypen sicherstellen
    for df in (df_train, df_val, df_test):
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["rating"]  = df["rating"].astype(float)

    # Trainset bauen
    trainset = build_trainset(df_train)

    best_params = {"n_factors": 100, "n_epochs": 25, "lr_all": 0.005, "reg_all": 0.02}
    if args.grid:
        print("Starte kleine Grid-Suche (3-fold CV auf Train, Metrik RMSE) ...")
        param_grid = {
            "n_factors": [50, 100],
            "n_epochs":  [20, 30],
            "lr_all":    [0.002, 0.005],
            "reg_all":   [0.02, 0.08],
        }
        data_for_cv = to_surprise_data(df_train)
        gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, joblib_verbose=0, n_jobs=-1)
        gs.fit(data_for_cv)
        best_params = gs.best_params["rmse"]
        print("Beste Params:", best_params, " (best RMSE=", gs.best_score["rmse"], ")")

    # Finales Modell mit besten Params trainieren
    algo = SVD(random_state=args.seed, **best_params)
    algo.fit(trainset)

    # Auswertung
    metrics_val  = evaluate(algo, trainset, df_val,  "VAL")
    metrics_test = evaluate(algo, trainset, df_test, "TEST")

    # Speichern
    model_path = MODELS / "cf_svd.pkl"
    joblib.dump(algo, model_path)
    meta = {
        "algo": "SVD",
        "best_params": best_params,
        "seed": args.seed,
        "metrics": {"val": metrics_val, "test": metrics_test},
        "train_n": int(len(df_train)),
        "val_n":   int(len(df_val)),
        "test_n":  int(len(df_test)),
    }
    with open(MODELS / "cf_svd_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Gespeichert: {model_path}")
    print("Meta:", json.dumps(meta, indent=2))
    print("Fertig. ✅")

if __name__ == "__main__":
    main()
