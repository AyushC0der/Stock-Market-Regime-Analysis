"""
Unit 2 - Ridge regression.

Syllabus topics covered here:
    - Ridge regression

Ridge regression adds an L2 penalty on the weights to the least-squares
objective:

    minimize  ||y - Xw||^2 + lambda * ||w||^2

which has the closed-form solution

    w_ridge = (X^T X + lambda * I)^-1 X^T y

As lambda -> 0, this collapses back to the plain least-squares solution
from least_squares.py. As lambda -> infinity, w is squeezed toward 0
(hence "shrinkage" - the classic ridge path plot below shows every
coefficient sliding toward zero as lambda grows). Crucially, this is NOT
just an ad-hoc trick: it is exactly the MAP estimate you get from putting
a zero-mean Gaussian prior on w and combining it with a Gaussian
likelihood - bayesian_linear_regression.py derives that connection
explicitly and shows the posterior MEAN equals this ridge solution for a
matching lambda.

We pick lambda using the VALIDATION split (never touching test), which is
the proper analogue of cross-validation once you've committed to a
chronological split.

Run: python -m src.unit2.regression.ridge_regression   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    REGRESSION_TARGET,
)

RESULTS_DIR = os.path.join("results", "unit2", "regression")

ALPHA_GRID = np.logspace(-3, 5, 33)  # lambda values to sweep, log-spaced


def ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """
    w = (X^T X + alpha * I) ^ -1 X^T y, fit on already-centered/standardized
    features so the intercept is just mean(y) and doesn't need its own
    regularization term.
    """
    n_features = X.shape[1]
    XtX = X.T @ X
    ridge_term = alpha * np.eye(n_features)
    w = np.linalg.solve(XtX + ridge_term, X.T @ y)
    return w


def verify_closed_form_matches_sklearn(X_train, y_train, alpha=1.0):
    y_centered = y_train - y_train.mean()
    w_manual = ridge_closed_form(X_train, y_centered, alpha)

    sk_model = Ridge(alpha=alpha, fit_intercept=True)
    sk_model.fit(X_train, y_train)

    max_diff = np.abs(w_manual - sk_model.coef_).max()
    print(f"Verifying closed-form ridge solution matches sklearn's Ridge (alpha={alpha}):")
    print(f"  max |w_manual - w_sklearn| = {max_diff:.10f}  (should be ~0)")


def shrinkage_path(X_train, y_train, feature_names):
    coef_paths = np.array([
        Ridge(alpha=alpha).fit(X_train, y_train).coef_ for alpha in ALPHA_GRID
    ])  # shape: (n_alphas, n_features)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(feature_names):
        ax.plot(ALPHA_GRID, coef_paths[:, i], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("lambda (regularization strength)")
    ax.set_ylabel("coefficient value")
    ax.set_title("Ridge shrinkage path: every coefficient slides toward 0 as lambda grows")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ridge_shrinkage_path.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def select_alpha_on_validation(splits):
    val_rmses = []
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha).fit(splits["X_train"], splits["y_train"])
        preds = model.predict(splits["X_val"])
        val_rmses.append(np.sqrt(mean_squared_error(splits["y_val"], preds)))

    best_idx = int(np.argmin(val_rmses))
    best_alpha = ALPHA_GRID[best_idx]
    print(f"\nSelecting lambda using the VALIDATION split (test stays untouched):")
    print(f"  best lambda = {best_alpha:.4f}  (val RMSE = {val_rmses[best_idx]:.6f})")
    if best_idx == len(ALPHA_GRID) - 1:
        print(
            "  note: validation error is still improving at the largest lambda we tried - "
            "that means the validation set finds essentially no linear signal worth keeping "
            "(it prefers shrinking weights toward 0, i.e. toward 'just predict the mean'), "
            "which is consistent with next-day returns being very hard to predict linearly."
        )
    return best_alpha, val_rmses


def compare_ridge_vs_ols_on_test(splits, best_alpha):
    ols = LinearRegression().fit(splits["X_train"], splits["y_train"])
    ridge = Ridge(alpha=best_alpha).fit(splits["X_train"], splits["y_train"])

    print("\nFinal test-set comparison (lambda chosen on val, evaluated once on test):")
    for name, model in [("OLS (lambda=0)", ols), (f"Ridge (lambda={best_alpha:.3f})", ridge)]:
        preds = model.predict(splits["X_test"])
        rmse = np.sqrt(mean_squared_error(splits["y_test"], preds))
        print(f"  {name:<22}: test RMSE = {rmse:.6f}, ||w|| = {np.linalg.norm(model.coef_):.5f}")


def main():
    print("Ridge regression (Unit 2)")
    df = load_model_ready_dataset()
    splits = get_model_ready_splits(df, REGRESSION_TARGET)
    print_split_sizes(splits, "Ridge regression")

    verify_closed_form_matches_sklearn(splits["X_train"], splits["y_train"], alpha=1.0)
    shrinkage_path(splits["X_train"], splits["y_train"], splits["feature_cols"])
    best_alpha, _ = select_alpha_on_validation(splits)
    compare_ridge_vs_ols_on_test(splits, best_alpha)


if __name__ == "__main__":
    main()
