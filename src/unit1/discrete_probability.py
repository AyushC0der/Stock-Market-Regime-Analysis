"""
Unit 1 - Probability theory for discrete random variables.

Syllabus topics covered here:
    - Discrete random variables
    - Fundamental rules of probability (sum rule, product rule)
    - Bayes' rule
    - Independence and conditional independence

Everything below uses two discretized variables built from real market
data instead of coin flips or dice, so the same three rules you'd see in
a textbook (sum rule, product rule, Bayes' rule) are demonstrated on
something you already understand:

    D  = Target_Direction        in {0 = down, 1 = up}          (Bernoulli RV)
    R  = RSI_Regime              in {Oversold, Neutral, Overbought}
    V  = Volatility_Regime       in {Low, High}   (used for conditional independence)

Run: python -m src.unit1.discrete_probability   (or via run_unit1.py)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("results", "unit1", "probability")


def load_all_tickers() -> pd.DataFrame:
    """Pool every ticker's feature file into one long table for bigger sample sizes."""
    frames = []
    if not os.path.isdir(PROCESSED_DIR):
        raise FileNotFoundError("data/processed/ not found. Run `python main.py` first.")
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        if fname.endswith("_features.csv"):
            df = pd.read_csv(os.path.join(PROCESSED_DIR, fname), parse_dates=["Date"])
            df["Ticker"] = fname[: -len("_features.csv")]
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No *_features.csv files found. Run `python main.py` first.")
    return pd.concat(frames, ignore_index=True)


def add_discrete_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket two continuous indicators into named categorical regimes."""
    df = df.copy()
    df["RSI_Regime"] = pd.cut(
        df["RSI"], bins=[-np.inf, 30, 70, np.inf], labels=["Oversold", "Neutral", "Overbought"]
    )
    # Median split on 20-day volatility -> "Low" vs "High" regime.
    vol_median = df["Volatility_20D"].median()
    df["Volatility_Regime"] = np.where(df["Volatility_20D"] <= vol_median, "Low", "High")
    return df.dropna(subset=["RSI_Regime"])


def fundamental_rules_demo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the joint distribution table P(D, R) and verify, numerically,
    the two rules everything else in probability theory is built from:

        Sum rule:      P(D=d) = sum_r P(D=d, R=r)
        Product rule:  P(D=d, R=r) = P(D=d | R=r) * P(R=r)
    """
    joint_counts = pd.crosstab(df["Target_Direction"], df["RSI_Regime"])
    joint_prob = joint_counts / joint_counts.values.sum()

    print("\nJoint distribution P(Direction, RSI_Regime):")
    print(joint_prob.round(4))

    # --- Sum rule: marginalize the joint over RSI_Regime ---
    marginal_direction_from_joint = joint_prob.sum(axis=1)
    marginal_direction_direct = df["Target_Direction"].value_counts(normalize=True).sort_index()
    print("\nSum rule check: P(Direction) from marginalizing the joint")
    print(marginal_direction_from_joint.round(4))
    print("...matches the direct empirical marginal:")
    print(marginal_direction_direct.round(4))
    assert np.allclose(
        marginal_direction_from_joint.values, marginal_direction_direct.values, atol=1e-9
    ), "Sum rule failed to reconcile - this should never happen with a plain crosstab."

    # --- Product rule: P(D, R) = P(D | R) * P(R) ---
    marginal_regime = df["RSI_Regime"].value_counts(normalize=True)
    cond_direction_given_regime = pd.crosstab(
        df["Target_Direction"], df["RSI_Regime"], normalize="columns"
    )
    reconstructed_joint = cond_direction_given_regime.mul(marginal_regime, axis=1)
    print("\nProduct rule check: P(D|R) * P(R) reconstructs the joint P(D, R)?")
    print(f"  max absolute difference from true joint: "
          f"{(reconstructed_joint - joint_prob).abs().values.max():.6f}  (should be ~0)")

    return joint_prob


def bayes_rule_demo(df: pd.DataFrame):
    """
    We directly observe P(Direction | RSI_Regime) from the data (easy:
    it's just a conditional frequency count). Bayes' rule lets us flip
    it around to P(RSI_Regime | Direction) without recounting anything -
    purely from P(Direction | RSI_Regime), P(RSI_Regime), and P(Direction):

        P(R | D) = P(D | R) * P(R) / P(D)
    """
    p_regime = df["RSI_Regime"].value_counts(normalize=True)
    p_direction_given_regime = pd.crosstab(
        df["Target_Direction"], df["RSI_Regime"], normalize="columns"
    )
    p_direction = df["Target_Direction"].value_counts(normalize=True)

    # Bayes' rule, applied by hand for Direction = 1 (up):
    numerator = p_direction_given_regime.loc[1] * p_regime
    p_regime_given_up = numerator / p_direction.loc[1]

    # Ground truth for comparison: just count it directly.
    p_regime_given_up_direct = (
        df.loc[df["Target_Direction"] == 1, "RSI_Regime"].value_counts(normalize=True)
    )

    print("\nBayes' rule: P(RSI_Regime | Direction = up), computed via Bayes' rule")
    print(p_regime_given_up.sort_index().round(4))
    print("...vs. computed directly from the data (should match):")
    print(p_regime_given_up_direct.sort_index().round(4))

    max_diff = (p_regime_given_up.sort_index() - p_regime_given_up_direct.sort_index()).abs().max()
    print(f"  max absolute difference: {max_diff:.6f}  (should be ~0)")


def independence_test(df: pd.DataFrame, col_a: str, col_b: str, label: str) -> float:
    """
    Two RVs A, B are independent iff P(A, B) = P(A) * P(B) for every
    combination of values. We can't check floating-point equality on
    noisy real-world counts, so instead we run a chi-square test of
    independence: it compares the observed joint counts against the
    counts we'd *expect* under independence, and returns a p-value.

    A large p-value (> 0.05, conventionally) means "the data is
    consistent with independence" - it does NOT prove independence, it
    just means we failed to find evidence against it.
    """
    contingency = pd.crosstab(df[col_a], df[col_b])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    verdict = "independent (fail to reject H0)" if p_value > 0.05 else "NOT independent (reject H0)"
    print(f"\nIndependence test: {label}")
    print(f"  chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_value:.4f} -> {verdict}")
    return p_value


def conditional_independence_demo(df: pd.DataFrame):
    """
    Check whether Direction and RSI_Regime look independent overall, and
    then again *within each Volatility_Regime stratum separately*. This is
    the concept of conditional independence: D independent of R could hold, be broken,
    or even flip once you condition on a third variable V - which is
    exactly why "correlation in the pooled data" claims are risky without
    checking what happens once you stratify.
    """
    print("\n=== Conditional independence: does Direction independent of RSI_Regime | Volatility_Regime? ===")
    independence_test(df, "Target_Direction", "RSI_Regime", "Direction vs RSI_Regime (pooled, unconditional)")

    for regime in sorted(df["Volatility_Regime"].unique()):
        subset = df[df["Volatility_Regime"] == regime]
        independence_test(
            subset, "Target_Direction", "RSI_Regime",
            f"Direction vs RSI_Regime | Volatility_Regime = {regime} (n={len(subset)})",
        )


def main():
    print("Discrete probability theory (Unit 1)")
    df = add_discrete_regimes(load_all_tickers())

    fundamental_rules_demo(df)
    bayes_rule_demo(df)
    conditional_independence_demo(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "discrete_probability_summary.txt")
    with open(summary_path, "w") as f:
        f.write("See console output / re-run this module to regenerate these numbers.\n")
        f.write(f"Rows analyzed: {len(df)} (pooled across all tickers)\n")
    print(f"\n  wrote {summary_path}")


if __name__ == "__main__":
    main()
