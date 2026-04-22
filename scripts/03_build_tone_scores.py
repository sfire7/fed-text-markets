import pandas as pd
from pathlib import Path

HAWKISH = [
    "inflation", "inflationary", "inflation expectations", "price stability",
    "tightening", "tighten", "tightened", "tightens", "firm", "strong", "robust", "solid", "strengthening", "strengthen", "elevated", "elevate", "elevating", "elevates", "persistent", "persistently", "persistence", "strong", "further", "upside", "pressures"
]

DOVISH = [
    "slowdown", "easing", "weaker", "moderation", "downside", "softening", "uncertainty", "decline", "accomodative", "slowing", 
]

def count_words(text, word_list):
    words = text.split()
    return sum(words.count(word) for word in word_list)

def main():
    input_path = Path("data/processed/fed_statements_clean.csv")
    output_path = Path("data/processed/tone_scores.csv")

    df = pd.read_csv(input_path)
    df = df[df["date"] != "2025-08-22"] # I had to remove this row, since it was not a standard FOMC statement and was skewing the tone scores.

    print("Rows loaded:", len(df))

    df["hawkish_count"] = df["clean_text"].apply(lambda x: count_words(str(x), HAWKISH))
    df["dovish_count"] = df["clean_text"].apply(lambda x: count_words(str(x), DOVISH))
    df["tone_score"] = df["hawkish_count"] - df["dovish_count"]
    df["tone_score_norm"] = df["tone_score"] / df["word_count"].replace(0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved tone scores to {output_path.resolve()}")
    print(df[["date", "title", "hawkish_count", "dovish_count", "tone_score", "tone_score_norm"]].head())

if __name__ == "__main__":
    main()

df = pd.read_csv("data/processed/tone_scores.csv")
df[["date", "hawkish_count", "dovish_count", "tone_score", "tone_score_norm"]].head()
df["tone_score"].describe()
df["tone_score_norm"].describe()
df[["date", "title", "hawkish_count", "dovish_count", "tone_score"]].sort_values("tone_score").head(10)
df[["date", "title", "hawkish_count", "dovish_count", "tone_score"]].sort_values("tone_score", ascending=False).head(10)