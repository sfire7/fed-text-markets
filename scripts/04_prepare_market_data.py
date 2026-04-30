import pandas as pd
from pathlib import Path

def main():
    input_path = Path("data/raw/market_data_raw.csv")
    output_path = Path("data/processed/market_data_clean.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    print("Rows loaded:", len(df))
    print("Columns found:", df.columns.tolist())

    required_cols= ["date", "us2yield", "sp500", "usd_index"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    market_cols = ["us2yield", "sp500", "usd_index"]

    for col in market_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nMissing values before dropping incomplete rows:")
    print(df.isna().sum())

    df = df.dropna(subset=["date", "us2yield", "sp500", "usd_index"]).copy()

    print("\nRows after dropping incomplete market rows:", len(df))

    df = df.sort_values("date").reset_index(drop=True)

    df["us2yield_change"] = df["us2yield"].diff()
    df["sp500_return"] = df["sp500"].pct_change(fill_method=None)
    df["usd_index_change"] = df["usd_index"].pct_change(fill_method=None)

    print("\nMissing values after calculating changes:")
    print(df.isna().sum())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSummary statistics:")
    print(
        df[
            [
                "us2yield",
                "sp500",
                "usd_index",
                "us2yield_change",
                "sp500_return",
                "usd_index_change"
            ]
        ].describe()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved cleaned market data to {output_path.resolve()}")
    print(df.head())


if __name__ == "__main__":
    main()