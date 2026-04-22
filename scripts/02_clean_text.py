import pandas as pd
import re
from pathlib import Path

def clean_text(text):
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

    df = pd.read_csv(input_path)

    print("Rows loaded:", len(df))
    print("Columns found:", df.columns.tolist())

    df["clean_text"] = df["statement_text"].apply(clean_text)
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned data to {output_path.resolve()}")
    print(df[["date", "title", "word_count"]].head())

if __name__ == "__main__":
    main()


df = pd.read_csv("data/raw/fed_statements_raw.csv")
len(df)
df.head()
df["date"].nunique()
df["url"].nunique()
df[["date", "url"]].head(10)

