import pandas as pd
'''
tr = pd.read_parquet("data/processed/ratings_train.parquet")
va = pd.read_parquet("data/processed/ratings_val.parquet")
te = pd.read_parquet("data/processed/ratings_test.parquet")

print(len(tr), len(va), len(te))
print(tr[["userId","movieId"]].duplicated().sum())  # sollte 0 sein
print(set(tr.userId) & set(va.userId))  # Überschneidung der Nutzer ist OK

'''
# kurzer Check, ob das Modell lädt und vorhersagt:
import joblib, pandas as pd
from surprise import Reader, Dataset

algo = joblib.load("models/cf_svd.pkl")
# Beispiel-Paar (existierend aus Train):
df = pd.read_parquet("data/processed/ratings_train.parquet")
u, i = int(df.iloc[0].userId), int(df.iloc[0].movieId)
print(algo.predict(u, i))  # Prediction(uid=..., iid=..., r_ui=None, est=..., ...)
