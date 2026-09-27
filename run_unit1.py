"""
Unit 1 runner: polynomial curve fitting + probability theory, all applied
to the processed stock data from Milestone 1.

Covers, in order:
    1. Polynomial curve fitting          (what/why ML, supervised learning, over/underfitting)
    2. Discrete probability + Bayes'     (discrete RVs, sum/product rule, Bayes' rule,
                                           independence, conditional independence)
    3. Continuous probability             (continuous RVs, densities, quantiles,
                                           mean/variance, expectation)
    4. Expectation & covariance            (multivariate expectation/covariance,
                                           portfolio variance, diversification)

Requires data/processed/*_features.csv to already exist - run `python
main.py` first if you haven't.

Run: python run_unit1.py
Outputs: figures + text summaries under results/unit1/
"""

from src.unit1 import (
    polynomial_curve_fitting,
    discrete_probability,
    continuous_probability,
    covariance_expectation,
)


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_unit1():
    section("1/4 - Polynomial curve fitting")
    polynomial_curve_fitting.main()

    section("2/4 - Discrete probability theory + Bayes' rule")
    discrete_probability.main()

    section("3/4 - Continuous probability theory")
    continuous_probability.main()

    section("4/4 - Expectation & covariance across tickers")
    covariance_expectation.main()

    section("Unit 1 complete")
    print("All figures and summaries were saved under results/unit1/.")
    print("Next: Unit 2 (linear models for regression & classification).")


if __name__ == "__main__":
    run_unit1()
