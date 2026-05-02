import pandas as pd
import re
from pathlib import Path


# Original / smaller dictionary
BASELINE_HAWKISH = [ "inflation", "inflationary", "inflation expectations", "price stability", "tightening", "tighten", "tightened", "tightens", "firm", "strong", "robust", "solid", "strengthening", "strengthen", "elevated", "persistent", "persistently", "persistence", "upside", "pressures" ]

BASELINE_DOVISH = [ "slowdown", "easing", "weaker", "moderation", "downside", "softening", "uncertainty", "decline", "accommodative", "slowing"]


# Expanded dictionary grouped around monetary-policy themes
EXPANDED_HAWKISH = [
    # Inflation pressure
    "inflation", "inflationary", "price pressures", "inflation expectations", "price stability", "elevated inflation", "persistent inflation", "upside risks", "inflation remains elevated",

    # Tightening / restrictive policy
    "tightening", "tighten", "tightened", "tightens", "restrictive", "policy firming", "further firming", "rate increases", "higher rates", "additional policy firming",

    # Strong economy / labour market
    "strong", "robust", "solid", "strengthening", "strong job gains", "labor market remains tight", "strong labor market", "low unemployment"
]

EXPANDED_DOVISH = [
    # Economic weakness / slowdown
    "slowdown", "slowing", "weaker", "softening", "moderation", "decline", "downside risks", "weaker growth", "economic activity slowed",

    # Easing / accommodative policy
    "easing", "ease", "accommodative", "lower rates", "rate cuts", "policy accommodation",

    # Uncertainty / financial stress
    "uncertainty", "financial strains", "tight credit conditions", "labor market has softened", "unemployment has risen"
]


EXCLUDED_DATES = ["2025-08-22"]


def count_terms(text, term_list):
    """
    Count both single-word and multi-word terms in cleaned statement text.
    """
    if pd.isna(text):
        return 0

    text = str(text)

    total = 0

    for term in term_list:
        pattern = rf"\b{re.escape(term)}\b"
        total += len(re.findall(pattern, text))

    return total


def build_tone_score(df, hawkish_terms, dovish_terms, label):
    """
    Create tone scores for one dictionary version.
    """
    out = df.copy()

    out[f"{label}_hawkish_count"] = out["clean_text"].apply(
        lambda x: count_terms(x, hawkish_terms)
    )

    out[f"{label}_dovish_count"] = out["clean_text"].apply(
        lambda x: count_terms(x, dovish_terms)
    )

    out[f"{label}_tone_score"] = (
        out[f"{label}_hawkish_count"] - out[f"{label}_dovish_count"]
    )

    out[f"{label}_tone_score_norm"] = (
        out[f"{label}_tone_score"] / out["word_count"].replace(0, pd.NA)
    )

    out[f"{label}_tone_score_norm"] = out[f"{label}_tone_score_norm"].fillna(0)

    return out


def main():
    input_path = Path("data/processed/fed_statements_clean.csv")
    output_path = Path("output/tables/dictionary_comparison.csv")
    detailed_output_path = Path("data/processed/tone_dictionary_comparison.csv")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    print("Rows loaded:", len(df))
    print("Columns found:", df.columns.tolist())

    required_cols = ["date", "title", "url", "clean_text", "word_count"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0)

    excluded_dates = pd.to_datetime(EXCLUDED_DATES)

    excluded = df[df["date"].isin(excluded_dates)]

    if not excluded.empty:
        print("\nExcluded non-standard statements:")
        print(excluded[["date", "title", "url"]])

    df = df[~df["date"].isin(excluded_dates)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    print("Rows after filtering:", len(df))

    # Build baseline and expanded scores
    df = build_tone_score(
        df,
        BASELINE_HAWKISH,
        BASELINE_DOVISH,
        "baseline"
    )

    df = build_tone_score(
        df,
        EXPANDED_HAWKISH,
        EXPANDED_DOVISH,
        "expanded"
    )

    # Compare whether the two dictionary versions move together
    dictionary_correlation = df[
        ["baseline_tone_score_norm", "expanded_tone_score_norm"]
    ].corr().iloc[0, 1]

    comparison = pd.DataFrame([
        {
            "dictionary": "baseline",
            "hawkish_terms": len(BASELINE_HAWKISH),
            "dovish_terms": len(BASELINE_DOVISH),
            "mean_tone_score_norm": df["baseline_tone_score_norm"].mean(),
            "std_tone_score_norm": df["baseline_tone_score_norm"].std(),
            "min_tone_score_norm": df["baseline_tone_score_norm"].min(),
            "max_tone_score_norm": df["baseline_tone_score_norm"].max(),
            "correlation_with_expanded": dictionary_correlation
        },
        {
            "dictionary": "expanded",
            "hawkish_terms": len(EXPANDED_HAWKISH),
            "dovish_terms": len(EXPANDED_DOVISH),
            "mean_tone_score_norm": df["expanded_tone_score_norm"].mean(),
            "std_tone_score_norm": df["expanded_tone_score_norm"].std(),
            "min_tone_score_norm": df["expanded_tone_score_norm"].min(),
            "max_tone_score_norm": df["expanded_tone_score_norm"].max(),
            "correlation_with_expanded": 1.0
        }
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_path, index=False)
    df.to_csv(detailed_output_path, index=False)

    print("\nDictionary comparison:")
    print(comparison)

    print(f"\nSaved summary comparison to {output_path.resolve()}")
    print(f"Saved detailed comparison data to {detailed_output_path.resolve()}")


if __name__ == "__main__":
    main()
