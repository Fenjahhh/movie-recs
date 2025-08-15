# scripts/split_ratings.py
# Zweck: ratings.csv user-weise in Train/Val/Test splitten und speichern

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def split_user_group(df_u: pd.DataFrame, rng: np.random.Generator,
                     train_p=0.8, val_p=0.1, test_p=0.1):
    """
    Teilt die Ratings eines einzelnen Users zufällig in Train/Val/Test.
    - Mindestens 1 Rating in Train, wenn möglich.
    - Bei sehr kleinen Usern (z.B. <3) fällt die Verteilung ggf. ungleichmäßiger aus.
    """
    m = len(df_u)
    if m == 0:
        return df_u.iloc[[]], df_u.iloc[[]], df_u.iloc[[]]

    # Shuffle Indizes reproduzierbar
    idx = np.arange(m)
    rng.shuffle(idx)

    # Zielanzahlen berechnen (grob), später an Kanten korrigieren
    n_train = int(round(train_p * m))
    n_val   = int(round(val_p   * m))
    n_test  = m - n_train - n_val

    # Sicherstellen, dass Train nicht leer ist (falls möglich)
    if n_train == 0 and m > 0:
        n_train, n_val = 1, max(0, n_val - 1)
        n_test = m - n_train - n_val

    # Kantenfälle ausgleichen (z.B. bei m=1/2/3)
    if n_val < 0: n_val = 0
    if n_test < 0: n_test = 0
    if n_train + n_val + n_test != m:
        n_test = m - n_train - n_val

    # Slices
    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train+n_val]
    idx_test  = idx[n_train+n_val:]

    return df_u.iloc[idx_train], df_u.iloc[idx_val], df_u.iloc[idx_test]

def split_all_users(ratings: pd.DataFrame, seed=42, train_p=0.8, val_p=0.1, test_p=0.1,
                    min_user_threshold=5):
    """
    Führt den user-weisen Split durch. Für User mit sehr wenigen Ratings (< min_user_threshold)
    nutzen wir eine einfache *globale* Zufallsverteilung, damit nichts kaputt geht.
    """
    rng = np.random.default_rng(seed)

    # Datentypen setzen
    ratings = ratings.copy()
    ratings["userId"] = ratings["userId"].astype("int32")
    ratings["movieId"] = ratings["movieId"].astype("int32")
    ratings["rating"] = ratings["rating"].astype("float32")

    # User mit ausreichend Daten
    by_user = ratings.groupby("userId", sort=False)
    train_parts, val_parts, test_parts = [], [], []

    small_users_rows = []  # Sammeln wir für den globalen Split
    for uid, df_u in by_user:
        if len(df_u) >= min_user_threshold:
            tr, va, te = split_user_group(df_u, rng, train_p, val_p, test_p)
            train_parts.append(tr)
            val_parts.append(va)
            test_parts.append(te)
        else:
            small_users_rows.append(df_u)

    # Große User wurden user-weise gesplittet
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings.iloc[[]].copy()
    val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else ratings.iloc[[]].copy()
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else ratings.iloc[[]].copy()

    # Kleine User global verteilen
    if len(small_users_rows) > 0:
        small_all = pd.concat(small_users_rows, ignore_index=True)
        m = len(small_all)
        idx = np.arange(m)
        rng.shuffle(idx)
        n_train = int(round(train_p * m))
        n_val   = int(round(val_p   * m))
        n_test  = m - n_train - n_val

        idx_train = idx[:n_train]
        idx_val   = idx[n_train:n_train+n_val]
        idx_test  = idx[n_train+n_val:]

        train_df = pd.concat([train_df, small_all.iloc[idx_train]], ignore_index=True)
        val_df   = pd.concat([val_df,   small_all.iloc[idx_val]],   ignore_index=True)
        test_df  = pd.concat([test_df,  small_all.iloc[idx_test]],  ignore_index=True)

    # Sanity-Checks: keine Überschneidungen
    def pairs(df): return set(zip(df.userId.astype(int), df.movieId.astype(int)))
    inter_train_val  = pairs(train_df) & pairs(val_df)
    inter_train_test = pairs(train_df) & pairs(test_df)
    inter_val_test   = pairs(val_df)   & pairs(test_df)

    assert len(inter_train_val) == 0,  f"Overlap train/val: {len(inter_train_val)}"
    assert len(inter_train_test) == 0, f"Overlap train/test: {len(inter_train_test)}"
    assert len(inter_val_test) == 0,   f"Overlap val/test: {len(inter_val_test)}"

    return train_df, val_df, test_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--min-user-threshold", type=int, default=5,
                    help="Ab wie vielen Ratings pro User machen wir user-weise Split (sonst global).")
    args = ap.parse_args()

    print("Lade ratings.csv ...")
    ratings_csv = RAW_DIR / "ratings.csv"
    ratings = pd.read_csv(ratings_csv)

    # Grobe Checks
    assert ratings[["userId","movieId","rating"]].isna().sum().sum() == 0, "NaNs in Kernspalten!"
    print("Anzahl Ratings gesamt:", len(ratings))

    print("Splitte ...")
    train_df, val_df, test_df = split_all_users(
        ratings,
        seed=args.seed,
        train_p=args.train,
        val_p=args.val,
        test_p=args.test,
        min_user_threshold=args.min_user_threshold
    )

    # Speichern (Parquet bevorzugt, sonst CSV)
    train_p = OUT_DIR / "ratings_train.parquet"
    val_p   = OUT_DIR / "ratings_val.parquet"
    test_p  = OUT_DIR / "ratings_test.parquet"

    def save_df(df, path_parquet, fallback_csv_name):
        try:
            df.to_parquet(path_parquet, index=False)
            print(f"Gespeichert: {path_parquet}")
            return str(path_parquet)
        except Exception as e:
            print(f"Parquet fehlgeschlagen ({e}). Fallback CSV.")
            csv_path = OUT_DIR / fallback_csv_name
            df.to_csv(csv_path, index=False)
            print(f"Gespeichert: {csv_path}")
            return str(csv_path)

    p_train = save_df(train_df, train_p, "ratings_train.csv")
    p_val   = save_df(val_df,   val_p,   "ratings_val.csv")
    p_test  = save_df(test_df,  test_p,  "ratings_test.csv")

    # Zusammenfassung
    summary = {
        "counts": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
            "total": len(ratings)
        },
        "unique_users": {
            "train": int(train_df["userId"].nunique()),
            "val":   int(val_df["userId"].nunique()),
            "test":  int(test_df["userId"].nunique()),
        },
        "unique_movies": {
            "train": int(train_df["movieId"].nunique()),
            "val":   int(val_df["movieId"].nunique()),
            "test":  int(test_df["movieId"].nunique()),
        },
        "paths": {
            "train": p_train,
            "val":   p_val,
            "test":  p_test
        },
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "seed": args.seed,
        "min_user_threshold": args.min_user_threshold
    }

    summary_path = OUT_DIR / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))
    print("Fertig. ✅")

if __name__ == "__main__":
    main()
