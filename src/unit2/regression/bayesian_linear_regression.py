"""
Unit 2 - Bayesian linear regression.

Syllabus topics covered here:
    - Bayesian linear regression

Instead of a single "best" weight vector w (what OLS/ridge give you),
Bayesian linear regression keeps a full PROBABILITY DISTRIBUTION over w,
updated from a prior using Bayes' rule (Unit 1!). With a Gaussian prior
    w ~ Normal(0, alpha^-1 I)
and a Gaussian likelihood
    y | x, w ~ Normal(w^T phi(x), beta^-1)
the posterior is available in closed form:
    S_N^-1 = alpha*I + beta * Phi^T Phi
    m_N    = beta * S_N * Phi^T y
and the predictive distribution for a new point x is also Gaussian:
    mean     = m_N^T phi(x)
    variance = 1/beta  +  phi(x)^T S_N phi(x)
                 ^^^^^^^^^^      ^^^^^^^^^^^^^^^^
              irreducible noise   uncertainty about w itself

That second variance term is the entire point of going Bayesian: it's
large far from the training data and shrinks as you collect more
evidence - point estimates (OLS/ridge) can't tell you that at all.

Two demonstrations:
    PART A - the classic Bishop-style toy: fit a cubic polynomial basis
    to a handful of noisy sine-wave points, and watch the predictive
    uncertainty band shrink as N grows, and stay wide in gaps between
    observations.
    PART B - the real thing: Bayesian linear regression on our actual
    stock features, predicting Target_Return, with calibrated predictive
    intervals evaluated on the untouched test set - and a direct check
    that the posterior MEAN equals the ridge regression solution for a
    matching lambda = alpha / beta.

Run: python -m src.unit2.regression.bayesian_linear_regression   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    REGRESSION_TARGET, RANDOM_SEED,
)

RESULTS_DIR = os.path.join("results", "unit2", "regression")


def posterior(Phi: np.ndarray, y: np.ndarray, alpha: float, beta: float):
    """Closed-form Gaussian posterior N(m_N, S_N) over weights."""
    n_features = Phi.shape[1]
    S_N_inv = alpha * np.eye(n_features) + beta * (Phi.T @ Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N @ Phi.T @ y
    return m_N, S_N


def predictive_distribution(Phi_new: np.ndarray, m_N: np.ndarray, S_N: np.ndarray, beta: float):
    """Per-point predictive mean and std: mean = m_N.phi(x), var = 1/beta + phi(x)^T S_N phi(x)."""
    mean = Phi_new @ m_N
    var = 1.0 / beta + np.einsum("ij,jk,ik->i", Phi_new, S_N, Phi_new)
    return mean, np.sqrt(var)


# ---------------------------------------------------------------------------
# PART A: classic toy demo, tying back to Unit 1's polynomial curve fitting
# ---------------------------------------------------------------------------

def polynomial_basis(x: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([x ** d for d in range(degree + 1)])


def toy_bayesian_demo(degree: int = 3, alpha: float = 2.0, beta: float = 25.0):
    rng = np.random.default_rng(RANDOM_SEED)
    x_dense = np.linspace(0, 1, 300)
    true_curve = np.sin(2 * np.pi * x_dense)
    Phi_dense = polynomial_basis(x_dense, degree)

    sample_sizes = [2, 5, 10, 25]
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(16, 4), sharey=True)

    for ax, n in zip(axes, sample_sizes):
        x_train = np.sort(rng.uniform(0, 1, n))
        y_train = np.sin(2 * np.pi * x_train) + rng.normal(0, 1 / np.sqrt(beta), n)
        Phi_train = polynomial_basis(x_train, degree)

        m_N, S_N = posterior(Phi_train, y_train, alpha, beta)
        mean, std = predictive_distribution(Phi_dense, m_N, S_N, beta)

        ax.plot(x_dense, true_curve, "g--", linewidth=1.3, label="true function")
        ax.fill_between(x_dense, mean - 2 * std, mean + 2 * std, alpha=0.25,
                         label="+/-2 std predictive band")
        ax.plot(x_dense, mean, color="tab:red", label="posterior predictive mean")
        ax.scatter(x_train, y_train, color="black", s=15, zorder=5)
        ax.set_title(f"N = {n} observations")
        ax.set_ylim(-2, 2)
        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle("Bayesian linear regression: predictive uncertainty shrinks as data accumulates")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "bayesian_toy_uncertainty_shrinks.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# PART B: the real thing, on actual stock features
# ---------------------------------------------------------------------------

def estimate_beta_from_residuals(X_train, y_train) -> float:
    """
    beta = 1 / noise variance. We don't know the true noise variance, so we
    estimate it from an OLS fit's residuals on the training data - a
    standard, simple plug-in estimate.
    """
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression().fit(X_train, y_train)
    residuals = y_train - ols.predict(X_train)
    residual_var = np.var(residuals, ddof=X_train.shape[1] + 1)
    return 1.0 / residual_var


def verify_posterior_mean_equals_ridge(X_train, y_train, alpha: float, beta: float):
    """
    The posterior mean m_N of Bayesian linear regression is mathematically
    identical to the ridge regression solution with lambda = alpha / beta.
    This is the concrete, numeric version of "regularization = a Gaussian
    prior in disguise."
    """
    Phi_train = np.column_stack([np.ones(len(X_train)), X_train])  # add bias/intercept column
    m_N, _ = posterior(Phi_train, y_train, alpha, beta)

    lam = alpha / beta
    # sklearn's Ridge objective is ||y - Xw||^2 + alpha_sklearn * ||w||^2 -
    # exactly the sum-of-squares form our posterior derivation uses, so
    # alpha_sklearn = lambda = alpha/beta reproduces the identical objective
    # over the bias-augmented Phi (fit_intercept=False since bias is column 0).
    ridge = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(Phi_train, y_train)

    max_diff = np.abs(m_N - ridge.coef_).max()
    print(f"\nVerifying Bayesian posterior mean == ridge solution (lambda = alpha/beta = {lam:.5f}):")
    print(f"  max |m_N - w_ridge| = {max_diff:.8f}  (should be ~0)")


def real_data_bayesian_regression(splits):
    alpha = 2.0  # prior precision (inverse prior variance) on the weights
    beta = estimate_beta_from_residuals(splits["X_train"], splits["y_train"])
    print(f"\nEstimated noise precision beta = {beta:.2f} (i.e. noise std = {1/np.sqrt(beta):.5f})")

    Phi_train = np.column_stack([np.ones(len(splits["X_train"])), splits["X_train"]])
    Phi_test = np.column_stack([np.ones(len(splits["X_test"])), splits["X_test"]])

    m_N, S_N = posterior(Phi_train, splits["y_train"], alpha, beta)
    mean_pred, std_pred = predictive_distribution(Phi_test, m_N, S_N, beta)

    verify_posterior_mean_equals_ridge(splits["X_train"], splits["y_train"], alpha, beta)

    # Calibration check: for a well-calibrated 95% interval, ~95% of true
    # test targets should fall inside mean +/- 1.96*std.
    y_test = splits["y_test"]
    inside_95 = np.mean(np.abs(y_test - mean_pred) <= 1.96 * std_pred)
    rmse = np.sqrt(np.mean((y_test - mean_pred) ** 2))

    print("\nBayesian linear regression on real stock features (predicting Target_Return):")
    print(f"  test RMSE (posterior mean predictions) = {rmse:.6f}")
    print(f"  average predictive std                 = {std_pred.mean():.6f}")
    print(f"  fraction of test targets inside the 95% predictive interval = {inside_95:.1%}")
    print(
        "  (should be near 95% for a well-calibrated model - real returns have "
        "fatter tails than a Gaussian likelihood assumes, so don't be surprised "
        "if this comes in a bit below 95%: that's the same fat-tails story from "
        "Unit 1's continuous_probability module showing up again here.)"
    )
    return mean_pred, std_pred


def main():
    print("Bayesian linear regression (Unit 2)")

    print("\n-- Part A: classic toy demo (uncertainty shrinks as N grows) --")
    toy_bayesian_demo()

    print("\n-- Part B: real stock data --")
    df = load_model_ready_dataset()
    splits = get_model_ready_splits(df, REGRESSION_TARGET)
    print_split_sizes(splits, "Bayesian linear regression")
    real_data_bayesian_regression(splits)


if __name__ == "__main__":
    main()
