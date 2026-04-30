import pandas as pd
from pathlib import Path


def main():
    tone_path = Path("data/processed/tone_scores.csv")
    market_path = Path("data/processed/market_data_clean.csv")
    output_path = Path("data/processed/analysis_dataset.csv")

    if not tone_path.exists():
        raise FileNotFoundError(f"Input file not found: {tone_path}")

    if not market_path.exists():
        raise FileNotFoundError(f"Input file not found: {market_path}")

    tone_df = pd.read_csv(tone_path)
    market_df = pd.read_csv(market_path)

    print("Tone rows loaded:", len(tone_df))
    print("Market rows loaded:", len(market_df))

    print("Tone columns found:", tone_df.columns.tolist())
    print("Market columns found:", market_df.columns.tolist())

    required_tone_cols = [ "date", "title", "tone_score", "tone_score_norm", "tone_score_change", "tone_score_norm_change" ]

    required_market_cols = [ "date", "us2yield_change", "sp500_return", "usd_index_change" ]

    missing_tone_cols = [col for col in required_tone_cols if col not in tone_df.columns]
    missing_market_cols = [col for col in required_market_cols if col not in market_df.columns]

    if missing_tone_cols:
        raise ValueError(f"Missing required tone columns: {missing_tone_cols}")

    if missing_market_cols:
        raise ValueError(f"Missing required market columns: {missing_market_cols}")

    tone_df["date"] = pd.to_datetime(tone_df["date"], errors="coerce")
    market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce")

    print("\nMissing dates:")
    print("Tone missing dates:", tone_df["date"].isna().sum())
    print("Market missing dates:", market_df["date"].isna().sum())

    print("\nDuplicate date checks:")
    print("Duplicate tone dates:", tone_df["date"].duplicated().sum())
    print("Duplicate market dates:", market_df["date"].duplicated().sum())

    market_cols = [ "date", "us2yield_change", "sp500_return", "usd_index_change" ]
    analysis_df = tone_df.merge(
        market_df[market_cols],
        on="date",
        how="left",
        validate="1:1",
        indicator=True
    )

    print("\nMerge summary:")
    print(analysis_df["_merge"].value_counts())

    unmatched = analysis_df[analysis_df["_merge"] != "both"]

    if not unmatched.empty:
        print("\nUnmatched FOMC dates:")
        print(unmatched[["date", "title", "_merge"]])

    analysis_df = analysis_df.drop(columns=["_merge"])

    print("\nRows in merged dataset:", len(analysis_df))

    print("\nMissing market reaction values after merge:")
    print("Missing us2yield_change values:", analysis_df["us2yield_change"].isna().sum())
    print("Missing sp500_return values:", analysis_df["sp500_return"].isna().sum())
    print("Missing usd_index_change values:", analysis_df["usd_index_change"].isna().sum())

    print("\nPreview:")
    print(
        analysis_df[
            [ "date", "tone_score", "tone_score_norm", "tone_score_change", "tone_score_norm_change", "us2yield_change", "sp500_return", "usd_index_change" ]
        ].head(10)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_path, index=False)

    print(f"\nSaved merged analysis dataset to {output_path.resolve()}")


if __name__ == "__main__":
    main()