import pandas as pd
import re
from pathlib import Path

HAWKISH = [
    "inflation", "inflationary", "price stability", "tightening", "tighten", "tightened", "tightens", "firm", "strong", "robust", "solid", "strengthening", "strengthen", "elevated", "elevate", "elevating", "elevates", "persistent", "persistently", "persistence", "upside", "pressures"
]

DOVISH = [
    "slowdown", "easing", "weaker", "moderation", "downside", "softening", "uncertainty", "decline", "accommodative", "slowing", 
]

EXCLUDED_DATES = ["2025-08-22"] # This was a non-standard statement that skewed the tone scores, so I had to exclude it.

def count_terms (text, term_list):
    """
    Count dictionary terms in cleaned text.
    """
    if pd.isna(text):
        return 0
    
    text = str(text)

    total = 0

    for term in term_list:
        pattern = rf"\b{re.escape(term)}\b"
        total += len(re.findall(pattern, text))

    return total

def main():
    input_path = Path("data/processed/fed_statements_clean.csv")
    output_path = Path("data/processed/tone_scores.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    

    df = pd.read_csv(input_path)

    print("Rows loaded before filtering:", len(df))
    print("Columns found:", df.columns.tolist())

    required_cols = ["date", "title", "clean_text", "word_count"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0)

    excluded_dates = pd.to_datetime(EXCLUDED_DATES)

    excluded = df[df["date"].isin(excluded_dates)]

    if not excluded.empty:
        print("\nExcluded non-standard statements:")
        print(excluded[["date", "title", "url"]])

    df = df[~df["date"].isin(excluded_dates)].copy()

    print("Rows after filtering:", len(df))

    df["hawkish_count"] = df["clean_text"].apply(lambda x: count_terms(x, HAWKISH))

    df["dovish_count"] = df["clean_text"].apply(lambda x: count_terms(x, DOVISH))


    df["tone_score"] = df["hawkish_count"] - df["dovish_count"]
    df["tone_score_norm"] = (df["tone_score"] / df["word_count"].replace(0, pd.NA))

    df["tone_score_norm"] = df["tone_score_norm"].fillna(0)

    df = df.sort_values("date").reset_index(drop=True)

    df["tone_score_change"] = df["tone_score"].diff()
    df["tone_score_norm_change"] = df["tone_score_norm"].diff()



    print("\nTone score summary:")
    print(
        df[["hawkish_count", "dovish_count", "tone_score", "tone_score_norm"]].describe()
    )

    print("\nPreview:")
    print(
        df[["date", "title", "hawkish_count", "dovish_count", "tone_score", "tone_score_norm"]].head()
    )

    print("\nMost dovish statements:")
    print(
        df[["hawkish_count", "dovish_count", "tone_score", "tone_score_norm"]]
        .sort_values("tone_score")
        .head(10)
    )

    print("\nMost hawkish statements:")
    print(
        df[["date", "title", "hawkish_count", "dovish_count", "tone_score"]]
        .sort_values("tone_score", ascending=False)
        .head(10)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved tone scores to {output_path.resolve()}")


if __name__ == "__main__":
    main()