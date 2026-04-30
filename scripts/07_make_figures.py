import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def scatter_plot_with_fit(df, x_col, y_col, title, x_label, y_label, save_path):
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    if len(plot_df) < 2:
        print(f"Skipping {save_path.name}: not enough observations.")
        return

    x = plot_df[x_col]
    y = plot_df[y_col]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.grid(True)

    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color='red')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def main():
    input_path = Path("data/processed/analysis_dataset.csv")
    figures_dir = Path("output/figures") 
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    

    print("Rows loaded:", len(df))
    print("Columns found:", df.columns.tolist())

    required_cols = [
        "date",
        "tone_score_norm",
        "tone_score_norm_change",
        "us2yield_change",
        "sp500_return",
        "usd_index_change"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    saved_files = []

    # 1. Tone score distribution
    path = figures_dir / "tone_score_distribution.png"
    plt.figure(figsize=(10, 5))
    plt.hist(df["tone_score_norm"].dropna(), bins=12)
    plt.title("Distribution of Normalised Tone Scores")
    plt.xlabel("Normalised Tone Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_files.append(path.name)

    # 2. Tone score over time
    path = figures_dir / "tone_score_over_time.png"
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["tone_score_norm"])
    plt.title("Normalised Tone Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Normalised Tone Score")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_files.append(path.name)

    # 3. Change in tone score over time
    path = figures_dir / "tone_score_change_over_time.png"
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["tone_score_norm_change"])
    plt.title("Change in Normalised Tone Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Change in Normalised Tone Score")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_files.append(path.name)

    # Scatter plots with fitted lines
    scatter_specs = [
        ("tone_score_norm", "us2yield_change", "Tone Score vs 2Y Yield Change", "Tone Score (Normalised)", "2Y Yield Change", "tone_vs_us2yield_fit.png"),
        ("tone_score_norm", "sp500_return", "Tone Score vs S&P 500 Return", "Tone Score (Normalised)", "S&P 500 Return", "tone_vs_sp500_fit.png"),
        ("tone_score_norm", "usd_index_change", "Tone Score vs USD Index Change", "Tone Score (Normalised)", "USD Index Change", "tone_vs_usd_fit.png"),
        ("tone_score_norm_change", "us2yield_change", "Change in Tone Score vs 2Y Yield Change", "Change in Tone Score (Normalised)", "2Y Yield Change", "tone_change_vs_us2yield_fit.png"),
        ("tone_score_norm_change", "sp500_return", "Change in Tone Score vs S&P 500 Return", "Change in Tone Score (Normalised)", "S&P 500 Return", "tone_change_vs_sp500_fit.png"),
        ("tone_score_norm_change", "usd_index_change", "Change in Tone Score vs USD Index Change", "Change in Tone Score (Normalised)", "USD Index Change", "tone_change_vs_usd_fit.png"),
    ]

    for x_col, y_col, title, x_label, y_label, filename in scatter_specs:
        path = figures_dir / filename
        scatter_plot_with_fit(df, x_col, y_col, title, x_label, y_label, path)
        saved_files.append(path.name)

    print(f"\nSaved figures to {figures_dir.resolve()}")
    print("Files created:")
    for file in saved_files:
        print("-", file)


if __name__ == "__main__":
    main()
