"""
Unit 2 - Maximum likelihood estimation & least squares linear regression.

Syllabus topics covered here:
    - Maximum likelihood estimation, least squares

The core result this module exists to make concrete: if you assume

    Target_Return = w . x + epsilon,   epsilon ~ Normal(0, sigma^2)

then the maximum-likelihood estimate of w is EXACTLY the ordinary
least-squares solution - maximizing the Gaussian log-likelihood over w
reduces algebraically to minimizing sum of squared errors. We show this
two ways: (1) solve the normal equations directly, (2) fit sklearn's
LinearRegression, and confirm the coefficients match to numerical
precision. Then we fit the model for real, on real (leak-free,
chronologically split) stock features, predicting next-day return.

Run: python -m src.unit2.regression.least_squares   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    REGRESSION_TARGET, FEATURE_COLUMNS,
)

RESULTS_DIR = os.path.join("results", "unit2", "regression")


def normal_equations_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    The textbook closed-form least-squares solution:
        w = (X^T X)^-1 X^T y
    (with a bias/intercept column of ones prepended to X). This is the
    same w that maximizes the Gaussian likelihood of the data - MLE
    under Gaussian noise IS least squares, not merely "similar to" it.
    """
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    # lstsq is used instead of a literal matrix inverse for numerical
    # stability, but it solves the exact same normal equations.
    w, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)
    return w


def verify_mle_equals_ols(X_train: np.ndarray, y_train: np.ndarray):
    w_normal_eq = normal_equations_fit(X_train, y_train)

    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    w_sklearn = np.concatenate([[sk_model.intercept_], sk_model.coef_])

    max_diff = np.abs(w_normal_eq - w_sklearn).max()
    print("Verifying MLE (Gaussian likelihood) == least squares (normal equations) == sklearn:")
    print(f"  max |w_normal_equations - w_sklearn| = {max_diff:.10f}  (should be ~0)")
    return sk_model


def evaluate(model, X, y, label: str):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    baseline_rmse = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))
    print(f"  {label}: RMSE={rmse:.6f}  R^2={r2:.4f}  "
          f"(vs. predict-the-mean baseline RMSE={baseline_rmse:.6f})")
    return rmse, r2


def plot_coefficients(model: LinearRegression, feature_names, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(np.abs(model.coef_))[::-1]
    ax.barh([feature_names[i] for i in order], model.coef_[order])
    ax.set_xlabel("OLS coefficient (on standardized features)")
    ax.set_title("Least squares: which features move Target_Return, and which way")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("Least squares / MLE linear regression (Unit 2)")
    df = load_model_ready_dataset()
    splits = get_model_ready_splits(df, REGRESSION_TARGET)
    print_split_sizes(splits, "Least squares regression")

    model = verify_mle_equals_ols(splits["X_train"], splits["y_train"])

    print("\nPredicting next-day return (Target_Return):")
    evaluate(model, splits["X_train"], splits["y_train"], "train")
    evaluate(model, splits["X_val"], splits["y_val"], "val")
    evaluate(model, splits["X_test"], splits["y_test"], "test")
    print(
        "  note: R^2 close to 0 (or slightly negative) on held-out data is the "
        "expected, honest result for next-day stock returns - it means the "
        "model isn't finding a real linear signal beyond noise, which is "
        "consistent with the efficient market hypothesis, not a bug."
    )

    plot_coefficients(
        model, splits["feature_cols"],
        os.path.join(RESULTS_DIR, "ols_coefficients.png"),
    )
    return model, splits


if __name__ == "__main__":
    main()
