import pandas as pd
tr = pd.read_parquet("data/processed/ratings_train.parquet")
va = pd.read_parquet("data/processed/ratings_val.parquet")
te = pd.read_parquet("data/processed/ratings_test.parquet")

print(len(tr), len(va), len(te))
print(tr[["userId","movieId"]].duplicated().sum())  # sollte 0 sein
print(set(tr.userId) & set(va.userId))  # Überschneidung der Nutzer ist OK
