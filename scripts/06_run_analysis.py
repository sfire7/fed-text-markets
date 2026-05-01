import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def significance_stars(p_value):
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.10:
        return "*"
    else:
        return ""
    
def run_model(df, y_var, x_vars):
    model_df = df[[y_var] + x_vars].dropna()

    X = model_df[x_vars]
    X = sm.add_constant(X)
    y = model_df[y_var]

    model = sm.OLS(y, X).fit()
    return model, model_df

def main():
    input_path = Path("data/processed/analysis_dataset.csv")
    output_dir = Path("output/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    print("Rows loaded:", len(df))

    # Summary statistics for key variables:
    
    summary_vars = ["tone_score", "tone_score_norm", "tone_score_change", "tone_score_norm_change", "us2yield_change", "sp500_return", "usd_index_change"]

    summary_stats = df[summary_vars].describe().round(4)
    summary_stats.to_csv(output_dir / "summary_stats.csv", float_format="%.4f")

    summary_table = summary_stats.T.reset_index()
    summary_table = summary_table.rename(columns={
        "index": "variable",
        "count": "n",
        "std": "standard_deviation",
        "min": "minimum",
        "25%": "q1",
        "50%": "median",
        "75%": "q3",
        "max": "maximum"
    })

    summary_table = summary_table[
        ["variable", "n", "mean", "standard_deviation", "minimum", "q1", "median", "q3", "maximum"]
    ]

    summary_table["n"] = summary_table["n"].astype(int)

    summary_table.to_csv(output_dir / "summary_stats_table.csv", index=False, float_format="%.4f")

    print("Saved summary statistics.")
    print(summary_table.to_string(index=False))

    regression_results = []

    models = {
        "baseline_us2y": ("us2yield_change", ["tone_score_norm"]),
        "baseline_sp500": ("sp500_return", ["tone_score_norm"]),
        "baseline_usd": ("usd_index_change", ["tone_score_norm"]),
        "change_us2y": ("us2yield_change", ["tone_score_norm_change"]),
        "change_sp500": ("sp500_return", ["tone_score_norm_change"]),
        "change_usd": ("usd_index_change", ["tone_score_norm_change"])
}

    for model_name, (y_var, x_vars) in models.items():
        model, model_df = run_model(df, y_var, x_vars)

        print(f"\nRegression results for {model_name}")
        print(model.summary())

        for var in model.params.index:
            regression_results.append({
                "model_name": model_name,
                "dependent_variable": y_var,
                "variable": var,
                "coefficient": model.params[var],
                "std_error": model.bse[var],
                "t_stat":model.tvalues[var],
                "p_value": model.pvalues[var],
                "r_squared": model.rsquared,
                "n_obs": int(model.nobs)
            })
    results_df = pd.DataFrame(regression_results)

    # Rounded detailed regression output
    results_df = results_df.round({
        "coefficient": 4,
        "std_error": 4,
        "t_stat": 4,
        "p_value": 4,
        "r_squared": 4,
        "n_obs": 0
    })

    results_df.to_csv(output_dir / "regression_results.csv", index=False, float_format="%.4f")


    # Cleaner one-row-per-model regression table
    main_results = results_df[results_df["variable"] != "const"].copy()

    main_results["significance"] = main_results["p_value"].apply(significance_stars)

    main_results["coefficient_with_stars"] = (
        main_results["coefficient"].map(lambda x: f"{x:.4f}") + main_results["significance"]
    )

    regression_table = main_results[[
        "model_name",
        "dependent_variable",
        "variable",
        "coefficient_with_stars",
        "std_error",
        "t_stat",
        "p_value",
        "r_squared",
        "n_obs"
    ]]

    regression_table = regression_table.rename(columns={
        "model_name": "model",
        "dependent_variable": "dependent variable",
        "variable": "explanatory variable",
        "coefficient_with_stars": "coefficient",
        "std_error": "standard error",
        "t_stat": "t-statistic",
        "p_value": "p-value",
        "r_squared": "R-squared",
        "n_obs": "observations"
    })

    regression_table.to_csv(
        output_dir / "regression_summary_table.csv",
        index=False,
        float_format="%.4f"
    )

    print("\nClean regression summary table:")
    print(regression_table.to_string(index=False))

    print(f"\nSaved regression results to {(output_dir / 'regression_results.csv').resolve()}")

if __name__ == "__main__":
    main()

