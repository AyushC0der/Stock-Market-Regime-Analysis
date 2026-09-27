"""
Unit 2 - Robust linear regression.

Syllabus topics covered here:
    - Robust linear regression

Ordinary least squares minimizes SQUARED error, which means a single
extreme outlier (think: an earnings-day 20% gap, a flash crash, a fat-
finger print) can pull the whole fit toward itself, because its
contribution to the loss grows quadratically. Robust regression losses
(Huber's loss here) grow linearly instead of quadratically once the
residual passes a threshold delta, so a handful of extreme points can no
longer dominate the fit.

We demonstrate this two ways:
    1. On the real, unmodified training data (which already contains
       genuine outlier days - COVID crash, earnings surprises, etc.)
    2. In a CONTROLLED experiment where we deliberately inject synthetic
       extreme outliers into a copy of the training targets and watch
       how much each model's coefficients move - this isolates the
       robustness property from whatever the real data happens to contain.

Run: python -m src.unit2.regression.robust_regression   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    REGRESSION_TARGET, RANDOM_SEED,
)

RESULTS_DIR = os.path.join("results", "unit2", "regression")


def fit_ols_and_huber(X_train, y_train):
    ols = LinearRegression().fit(X_train, y_train)
    huber = HuberRegressor(epsilon=1.35).fit(X_train, y_train)  # epsilon=1.35 is Huber's classic default
    return ols, huber


def compare_on_real_data(splits):
    ols, huber = fit_ols_and_huber(splits["X_train"], splits["y_train"])
    print("\nOLS vs. Huber, both fit on the real (unmodified) training data:")
    for name, model in [("OLS", ols), ("Huber", huber)]:
        preds = model.predict(splits["X_test"])
        rmse = np.sqrt(mean_squared_error(splits["y_test"], preds))
        # Median absolute error is itself a robust metric - useful alongside RMSE,
        # which is dominated by whatever large errors exist in the test set.
        mae_median = np.median(np.abs(splits["y_test"] - preds))
        print(f"  {name:<6}: test RMSE={rmse:.6f}, median abs error={mae_median:.6f}")
    return ols, huber


def contamination_experiment(X_train, y_train, contamination_fracs=(0.0, 0.005, 0.02, 0.05)):
    """
    Fit OLS and Huber on the clean data, then repeatedly re-fit on a
    version of y_train where a growing fraction of points have been
    replaced with extreme synthetic outliers (10x a typical daily move).
    Track how far each model's coefficients move from their "clean" fit.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    clean_ols = LinearRegression().fit(X_train, y_train)
    clean_huber = HuberRegressor(epsilon=1.35).fit(X_train, y_train)
    clean_ols_coef = np.concatenate([[clean_ols.intercept_], clean_ols.coef_])
    clean_huber_coef = np.concatenate([[clean_huber.intercept_], clean_huber.coef_])

    typical_move = np.std(y_train)
    ols_drift, huber_drift = [], []

    for frac in contamination_fracs:
        y_contaminated = y_train.copy()
        n_outliers = int(frac * len(y_train))
        if n_outliers > 0:
            idx = rng.choice(len(y_train), size=n_outliers, replace=False)
            # Extreme synthetic shocks: +/-10 typical daily moves.
            y_contaminated[idx] = rng.choice([-1, 1], size=n_outliers) * 10 * typical_move

        ols = LinearRegression().fit(X_train, y_contaminated)
        huber = HuberRegressor(epsilon=1.35).fit(X_train, y_contaminated)
        ols_coef = np.concatenate([[ols.intercept_], ols.coef_])
        huber_coef = np.concatenate([[huber.intercept_], huber.coef_])

        ols_drift.append(np.linalg.norm(ols_coef - clean_ols_coef))
        huber_drift.append(np.linalg.norm(huber_coef - clean_huber_coef))

    print("\nContamination experiment: how far do coefficients move as we inject outliers?")
    print(f"  {'contamination':>13} | {'OLS coef drift':>15} | {'Huber coef drift':>17}")
    for frac, od, hd in zip(contamination_fracs, ols_drift, huber_drift):
        print(f"  {frac:>12.1%} | {od:>15.5f} | {hd:>17.5f}")
    print(
        "  -> OLS drift should grow much faster than Huber's as contamination "
        "increases: that's robust regression doing its job."
    )
    return contamination_fracs, ols_drift, huber_drift


def plot_contamination_drift(fracs, ols_drift, huber_drift):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fracs, ols_drift, "o-", label="OLS coefficient drift")
    ax.plot(fracs, huber_drift, "o-", label="Huber coefficient drift")
    ax.set_xlabel("fraction of training targets replaced with extreme outliers")
    ax.set_ylabel("L2 distance from the clean-data fit")
    ax.set_title("Robust regression: Huber's coefficients move far less under contamination")
    ax.legend()
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "robust_regression_contamination.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("Robust linear regression (Unit 2)")
    df = load_model_ready_dataset()
    splits = get_model_ready_splits(df, REGRESSION_TARGET)
    print_split_sizes(splits, "Robust regression")

    compare_on_real_data(splits)
    fracs, ols_drift, huber_drift = contamination_experiment(splits["X_train"], splits["y_train"])
    plot_contamination_drift(fracs, ols_drift, huber_drift)


if __name__ == "__main__":
    main()
