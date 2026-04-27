import pandas as pd
import re
from pathlib import Path

def clean_text(text):
    """
    Clean FOMC statement text for dictionary based tone analysis.
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def main():
    input_path = Path("data/raw/fed_statements_raw.csv")
    output_path = Path("data/processed/fed_statements_clean.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    print("Rows loaded:", len(df))
    print("Columns found:", df.columns.tolist())

    required_cols = ["date", "url", "title", "statement_text"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["clean_text"] = df["statement_text"].apply(clean_text)
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    print("Unique dates:", df["date"].nunique())
    print("Unique URLs:", df["url"].nunique())
    print("Missing dates:", df["date"].isna().sum())
    print("Missing statement text:", df["statement_text"].isna().sum())

    print("\nWord count statistics:")
    print(df["word_count"].describe())


    print("\nPreview:")
    print(df[["date", "title", "word_count"]].head())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned data to {output_path.resolve()}")

if __name__ == "__main__":
    main()



