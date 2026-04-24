import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def scatter_plot_with_fit(df, x_col, y_col, title, x_label, y_label, save_path):
    plot_df = df.dropna(subset=[x_col, y_col])

    x = plot_df[x_col]
    y = plot_df[y_col]

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)

    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color='red')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def main():
    input_path = Path("data/processed/analysis_dataset.csv")
    figures_dir = Path("output/figures") 
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    # Tone score distribution:
    plt.figure(figsize=(10, 5))
    plt.hist(df["tone_score_norm"].dropna(), bins=12)
    plt.title("Distribution of Normalised Tone Scores")
    plt.xlabel("Normalised Tone Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(figures_dir / "tone_score_distribution.png")
    plt.close()

    # Tone score over time:
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["tone_score_norm"])
    plt.title("Normalised Tone Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Normalised Tone Score")
    plt.tight_layout()
    plt.savefig(figures_dir / "tone_score_over_time.png")
    plt.close()

    # Change in normalised tone score over time:
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["tone_score_norm_change"])
    plt.title("Change in Normalised Tone Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Change in Normalised Tone Score")
    plt.tight_layout()
    plt.savefig(figures_dir / "tone_score_change_over_time.png")
    plt.close()

    # Scatter plots with fitted lines:
    scatter_plot_with_fit(df, "tone_score_norm", "us2yield_change", "Tone Score vs 2Y Yield Change", "Tone Score (Normalised)", "2Y Yield Change", figures_dir / "tone_vs_us2yield_fit.png")
    scatter_plot_with_fit(df, "tone_score_norm", "sp500_return", "Tone Score vs S&P 500 Return", "Tone Score (Normalised)", "S&P 500 Return", figures_dir / "tone_vs_sp500_fit.png")
    scatter_plot_with_fit(df, "tone_score_norm", "usd_index_change", "Tone Score vs USD Index Change", "Tone Score (Normalised)", "USD Index Change", figures_dir / "tone_vs_usd_fit.png")
    scatter_plot_with_fit(df, "tone_score_norm_change", "us2yield_change", "Change in Tone Score vs 2Y Yield Change", "Change in Tone Score (Normalised)", "2Y Yield Change", figures_dir / "tone_change_vs_us2yield_fit.png")
    scatter_plot_with_fit(df, "tone_score_norm_change", "sp500_return", "Change in Tone Score vs S&P 500 Return", "Change in Tone Score (Normalised)", "S&P 500 Return", figures_dir / "tone_change_vs_sp500_fit.png")
    scatter_plot_with_fit(df, "tone_score_norm_change", "usd_index_change", "Change in Tone Score vs USD Index Change", "Change in Tone Score (Normalised)", "USD Index Change", figures_dir / "tone_change_vs_usd_fit.png")
    

    print(f"Saved scatter plots with fitted lines to {figures_dir.resolve()}")

if __name__ == "__main__":
    main()

   