import pandas as pd
import numpy as np
from pathlib import Path

DATA_RAW = Path("../data/raw") if Path(".").name=="notebooks" else Path("data/raw")

ratings = pd.read_csv(DATA_RAW / "ratings.csv")
movies  = pd.read_csv(DATA_RAW / "movies.csv")

ratings.head(), ratings.shape, ratings.dtypes

# NaNs prüfen
ratings.isna().sum(), movies.isna().sum()

# Typen harmonisieren (IDs integer, rating float)
ratings = ratings.astype({"userId":"int32", "movieId":"int32", "rating":"float32", "timestamp":"int64"})
movies  = movies.astype({"movieId":"int32", "title":"string", "genres":"string"})

# Grundlegende Stats
ratings.describe(include="all")

# Gültige Rating-Skala prüfen (MovieLens small: 0.5..5.0 in 0.5-Schritten)
ratings["rating"].unique()[:10], ratings["rating"].min(), ratings["rating"].max()

# Doppelte Zeilen? (sollte es nicht geben)
dup_count = ratings.duplicated(subset=["userId","movieId"]).sum()
dup_count

# Anzahl Ratings pro User / pro Film
user_counts  = ratings.groupby("userId")["movieId"].count()
movie_counts = ratings.groupby("movieId")["userId"].count()

user_counts.describe(), movie_counts.describe()

# Plots - Verteilung der Ratings pro User / pro Film
import matplotlib.pyplot as plt

ratings["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Ratings-Verteilung"); plt.xlabel("Rating"); plt.ylabel("Anzahl"); plt.show()

user_counts.plot(kind="hist", bins=50)
plt.title("Ratings pro User"); plt.xlabel("#Ratings/User"); plt.show()

movie_counts.plot(kind="hist", bins=50)
plt.title("Ratings pro Film"); plt.xlabel("#Ratings/Film"); plt.show()

# Alle movieIds im ratings auch in movies vorhanden?
missing_movies = set(ratings["movieId"].unique()) - set(movies["movieId"].unique())
print(len(missing_movies))


# 1) Größe der DataFrames
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)

# 2) Erste Zeilen ansehen
print("\n=== Movies head ===")
print(movies.head())

print("\n=== Ratings head ===")
print(ratings.head())

# 3) Fehlende Werte zählen
print("\n=== Missing Values ===")
print("Movies:\n", movies.isna().sum())
print("Ratings:\n", ratings.isna().sum())

# 4) Datentypen
print("\n=== Data Types ===")
print("Movies:\n", movies.dtypes)
print("Ratings:\n", ratings.dtypes)

# 5) Min/Max Ratings
print("\n=== Ratings Range ===")
print(ratings["rating"].min(), "-", ratings["rating"].max())

# 6) Verteilung der Ratings (optional: nur Zählung)
print("\n=== Rating Value Counts ===")
print(ratings["rating"].value_counts().sort_index())

# 7) Beispiel: Genres-Spalten-Inhalt
print("\n=== Unique Genres Examples ===")
print(movies["genres"].unique()[:10])

# 8) Prüfen auf 'no genres listed'
print("\nNumber of movies with no genres listed:",
      (movies["genres"] == "(no genres listed)").sum())

# 9) Titel-Beispiele mit Jahr
print("\n=== Movie Titles Examples ===")
print(movies["title"].sample(10, random_state=42))

ratings.to_parquet("data/processed/ratings_raw.parquet", index=False)
movies.to_parquet("data/processed/movies_raw.parquet", index=False)


