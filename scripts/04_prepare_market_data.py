import pandas as pd
from pathlib import Path

def main():
    input_path = Path("data/raw/market_data_raw.csv")
    output_path = Path("data/processed/market_data_clean.csv")

    df = pd.read_csv(input_path)

    print("Rows loaded:", len(df))
    print("Columns found:", df.columns.tolist())

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])


    df["us2yield"] = pd.to_numeric(df["us2yield"], errors='coerce')
    df["sp500"] = pd.to_numeric(df["sp500"], errors='coerce')
    df["usd_index"] = pd.to_numeric(df["usd_index"], errors='coerce')


    df = df.sort_values("date").reset_index(drop=True)


    df["us2yield_change"] = df["us2yield"].diff()
    df["sp500_return"] = df["sp500"].pct_change()
    df["usd_index_change"] = df["usd_index"].pct_change()


    df = df.dropna(subset=["date", "us2yield", "sp500", "usd_index"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved cleaned market data to {output_path.resolve()}")
    print(df[["date", "us2yield", "sp500", "usd_index", "us2yield_change", "sp500_return", "usd_index_change"]].head())

if __name__ == "__main__":
    main()
    