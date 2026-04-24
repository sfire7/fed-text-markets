import pandas as pd
import statsmodels.api as sm
from pathlib import Path

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

    summary_stats = df[summary_vars].describe()
    summary_stats.to_csv(output_dir / "summary_stats.csv")

    print("Saved summary statistics.")
    print(summary_stats)


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
    results_df.to_csv(output_dir / "regression_results.csv", index=False)

    print(f"\nSaved regression results to {(output_dir / 'regression_results.csv').resolve()}")

if __name__ == "__main__":
    main()

