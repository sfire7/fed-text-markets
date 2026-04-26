import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


def scatter_plot_with_fit_interactive(
    df,
    x_col,
    y_col,
    title,
    x_label,
    y_label,
    save_path
):
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    x = plot_df[x_col]
    y = plot_df[y_col]

    m, b = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="FOMC observations",
            text=plot_df["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate=(
                "Date: %{text}<br>"
                + f"{x_label}: " + "%{x:.4f}<br>"
                + f"{y_label}: " + "%{y:.4f}<br>"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Fitted line",
            hovertemplate=(
                f"{x_label}: " + "%{x:.4f}<br>"
                + f"Predicted {y_label}: " + "%{y:.4f}<br>"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=520
    )

    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)


def main():
    input_path = Path("data/processed/analysis_dataset.csv")
    figures_dir = Path("output/interactive_figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    # 1. Tone score distribution
    fig = px.histogram(
        df.dropna(subset=["tone_score_norm"]),
        x="tone_score_norm",
        nbins=12,
        title="Distribution of Normalised Tone Scores",
        labels={
            "tone_score_norm": "Normalised Tone Score"
        }
    )

    fig.update_layout(
        yaxis_title="Frequency",
        template="plotly_white",
        height=520
    )

    fig.write_html(
        figures_dir / "tone_score_distribution.html",
        include_plotlyjs="cdn",
        full_html=True
    )

    # 2. Tone score over time
    fig = px.line(
        df,
        x="date",
        y="tone_score_norm",
        title="Normalised Tone Score Over Time",
        labels={
            "date": "Date",
            "tone_score_norm": "Normalised Tone Score"
        }
    )

    fig.update_traces(
        mode="lines+markers",
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Normalised Tone Score: %{y:.4f}<br>"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=520
    )

    fig.write_html(
        figures_dir / "tone_score_over_time.html",
        include_plotlyjs="cdn",
        full_html=True
    )

    # 3. Change in tone score over time
    fig = px.line(
        df,
        x="date",
        y="tone_score_norm_change",
        title="Change in Normalised Tone Score Over Time",
        labels={
            "date": "Date",
            "tone_score_norm_change": "Change in Normalised Tone Score"
        }
    )

    fig.update_traces(
        mode="lines+markers",
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Change in Tone Score: %{y:.4f}<br>"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=520
    )

    fig.write_html(
        figures_dir / "tone_score_change_over_time.html",
        include_plotlyjs="cdn",
        full_html=True
    )

    # 4. Scatter plots with fitted lines
    scatter_plot_with_fit_interactive(
        df,
        "tone_score_norm",
        "us2yield_change",
        "Tone Score vs 2Y Yield Change",
        "Tone Score (Normalised)",
        "2Y Yield Change",
        figures_dir / "tone_vs_us2yield_fit.html"
    )

    scatter_plot_with_fit_interactive(
        df,
        "tone_score_norm",
        "sp500_return",
        "Tone Score vs S&P 500 Return",
        "Tone Score (Normalised)",
        "S&P 500 Return",
        figures_dir / "tone_vs_sp500_fit.html"
    )

    scatter_plot_with_fit_interactive(
        df,
        "tone_score_norm",
        "usd_index_change",
        "Tone Score vs USD Index Change",
        "Tone Score (Normalised)",
        "USD Index Change",
        figures_dir / "tone_vs_usd_fit.html"
    )

    scatter_plot_with_fit_interactive(
        df,
        "tone_score_norm_change",
        "us2yield_change",
        "Change in Tone Score vs 2Y Yield Change",
        "Change in Tone Score (Normalised)",
        "2Y Yield Change",
        figures_dir / "tone_change_vs_us2yield_fit.html"
    )

    scatter_plot_with_fit_interactive(
        df,
        "tone_score_norm_change",
        "sp500_return",
        "Change in Tone Score vs S&P 500 Return",
        "Change in Tone Score (Normalised)",
        "S&P 500 Return",
        figures_dir / "tone_change_vs_sp500_fit.html"
    )

    scatter_plot_with_fit_interactive(
        df,
        "tone_score_norm_change",
        "usd_index_change",
        "Change in Tone Score vs USD Index Change",
        "Change in Tone Score (Normalised)",
        "USD Index Change",
        figures_dir / "tone_change_vs_usd_fit.html"
    )

    print(f"Saved interactive figures to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()