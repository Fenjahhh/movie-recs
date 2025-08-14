# scripts/prepare_movies.py
# Zweck: movies.csv säubern und als movies_processed.parquet speichern

import re
import json
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOVIES_CSV = RAW_DIR / "movies.csv"
OUT_PARQUET = OUT_DIR / "movies_processed.parquet"
OUT_CSV = OUT_DIR / "movies_processed.csv"       # optional als CSV
OUT_IDMAP = OUT_DIR / "id_map.json"

def extract_year(title: str):
    """
    Holt das 4-stellige Jahr am Ende des Titels, falls vorhanden.
    Beispiel: 'Se7en (1995)' -> 1995
              'Se7en (a.k.a. Seven) (1995)' -> 1995
    """
    if not isinstance(title, str):
        return None
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None

def strip_year(title: str):
    """
    Entfernt das letzte '(YYYY)' am Ende aus dem Titel.
    """
    if not isinstance(title, str):
        return title
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()

def parse_genres(genres: str):
    """
    Wandelt 'Action|Adventure|Sci-Fi' in ['Action','Adventure','Sci-Fi'] um.
    '(no genres listed)' -> []
    """
    if not isinstance(genres, str) or genres.strip() == "" or genres == "(no genres listed)":
        return []
    return [g.strip() for g in genres.split("|") if g.strip()]

def build_content_text(clean_title: str, year, genres_list):
    """
    Baut ein kompaktes Textfeld für Embeddings, z.B.:
    'Toy Story (1995) [Animation, Children, Comedy]'
    Regeln:
      - Jahr nur, wenn vorhanden
      - eckige Klammern nur, wenn Genres vorhanden
    """
    parts = [clean_title]
    if year is not None:
        parts[-1] = f"{parts[-1]} ({year})"
    if genres_list:
        parts.append(f"[{', '.join(genres_list)}]")
    return " ".join(parts)

def main():
    print("Lade movies.csv ...")
    df = pd.read_csv(MOVIES_CSV, encoding="utf-8")

    # Sicherheits-Umbenennungen, falls Spalten anderer Name hätten (sollten nicht nötig sein)
    df = df.rename(columns={"genres": "genres_raw"})
    
    # Typen setzen
    df["movieId"] = df["movieId"].astype("int32")
    df["title"] = df["title"].astype("string")
    df["genres_raw"] = df["genres_raw"].astype("string")

    # Jahr extrahieren & clean_title bauen
    print("Extrahiere Jahr & säubere Titel ...")
    df["year"] = df["title"].map(extract_year)
    df["clean_title"] = df["title"].map(strip_year)

    # Genres als Liste
    print("Verarbeite Genres ...")
    df["genres_list"] = df["genres_raw"].map(parse_genres)

    # content_text bauen
    print("Baue content_text ...")
    df["content_text"] = [
        build_content_text(ct, y, gl)
        for ct, y, gl in zip(df["clean_title"], df["year"], df["genres_list"])
    ]

    # Duplikate auf movieId sicherheitshalber entfernen (sollte es nicht geben)
    before = len(df)
    df = df.drop_duplicates(subset=["movieId"], keep="first")
    after = len(df)
    if after != before:
        print(f"Warnung: {before - after} Duplikate nach movieId entfernt.")

    # Relevante Spalten sortieren
    cols = ["movieId", "title", "clean_title", "year", "genres_raw", "genres_list", "content_text"]
    df = df[cols]

    # Sanity-Prints
    print("\n=== Beispiele (5 Zeilen) ===")
    print(df.head(5))
    print("\n=== 'no genres listed' Anzahl ===")
    print((df["genres_raw"] == "(no genres listed)").sum())
    print("\n=== content_text Beispiele ===")
    print(df["content_text"].sample(5, random_state=42))

    # Speichern
    print(f"\nSpeichere nach {OUT_PARQUET} ...")
    try:
        df.to_parquet(OUT_PARQUET, index=False)
    except Exception as e:
        print("Parquet fehlgeschlagen (fehlt pyarrow?). Fallback: CSV.")
        print("Fehler:", e)
        df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # ID-Map für später (Index ↔ movieId)
    print(f"Schreibe ID-Map nach {OUT_IDMAP} ...")
    id_map = {
        "row_to_movieId": df["movieId"].tolist(),
        "movieId_to_row": {int(mid): i for i, mid in enumerate(df["movieId"].tolist())},
    }
    with open(OUT_IDMAP, "w", encoding="utf-8") as f:
        json.dump(id_map, f)

    print("\nFertig. ✅")

if __name__ == "__main__":
    main()
