import pandas as pd
from pathlib import Path

def main():
    tone_path = Path("data/processed/tone_scores.csv")
    market_path = Path("data/processed/market_data_clean.csv")
    output_path = Path("data/processed/analysis_dataset.csv")

    tone_df = pd.read_csv(tone_path)
    market_df = pd.read_csv(market_path)

    tone_df["date"] = pd.to_datetime(tone_df["date"])
    market_df["date"] = pd.to_datetime(market_df["date"])

    analysis_df = tone_df.merge(market_df[["date", "us2yield_change", "sp500_return", "usd_index_change"]], on="date", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_path, index=False)

    print(f"Saved merged analysis dataset to {output_path.resolve()}")
    print("Rows in merged dataset:", len(analysis_df))
    print("Missing usd2yield_change values:", analysis_df["us2yield_change"].isna().sum())
    print("Missing sp500_return values:", analysis_df["sp500_return"].isna().sum())
    print("Missing usd_index_change values:", analysis_df["usd_index_change"].isna().sum())
    
    print(analysis_df[["date", "tone_score", "tone_score_norm", "us2yield_change", "sp500_return", "usd_index_change"]].head(10))

if __name__ == "__main__":
    main()

        