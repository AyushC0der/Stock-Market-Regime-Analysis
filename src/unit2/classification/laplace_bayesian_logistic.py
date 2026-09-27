"""
Unit 2 - Laplace approximation & Bayesian logistic regression.

Syllabus topics covered here:
    - Laplace approximation
    - Bayesian logistic regression

Plain logistic regression gives you one weight vector w_MAP (or w_MLE) and
treats it as the truth. Bayesian logistic regression instead wants the
full posterior P(w | data) - but unlike linear regression, there's no
closed form here (the sigmoid breaks the Gaussian conjugacy that made
Bayesian linear regression tractable). The LAPLACE APPROXIMATION is the
fix: approximate the posterior as a Gaussian centered at the mode (the
MAP estimate w_MAP), with covariance equal to the inverse of the
log-posterior's curvature (Hessian) at that mode:

    P(w | data)  ~=  Normal(w_MAP, H^-1)
    H = alpha*I + X^T R X       (R = diag(p_i (1-p_i)), same as IRLS)

Once we have that Gaussian approximation, predicting for a new point x
means integrating the sigmoid over the posterior instead of just
plugging in w_MAP - which has no closed form either, but a very good
closed-form APPROXIMATION exists (Bishop eq. 4.153):

    a           ~  Normal(mu_a, sigma_a^2)          where mu_a = w_MAP . x,
                                                     sigma_a^2 = x^T H^-1 x
    P(C1 | x)   ~=  sigmoid( kappa(sigma_a^2) * mu_a ),
                  kappa(sigma_a^2) = 1 / sqrt(1 + pi * sigma_a^2 / 8)

The effect: kappa <= 1 always, so the Bayesian predictive probability is
always pulled TOWARD 0.5 relative to the naive plug-in prediction
sigmoid(mu_a) - more so exactly where the model is less certain about w
(large sigma_a^2, e.g. far from the training data, or near the decision
boundary). This directly fixes logistic regression's tendency to be
overconfident.

Run: python -m src.unit2.classification.laplace_bayesian_logistic   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    CLASSIFICATION_TARGET, RANDOM_SEED,
)

RESULTS_DIR = os.path.join("results", "unit2", "classification")

PRIOR_PRECISION_ALPHA = 1.0  # alpha: prior precision on the weights (excludes bias)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def map_logistic_regression(X: np.ndarray, y: np.ndarray, alpha: float = PRIOR_PRECISION_ALPHA, n_iter: int = 50):
    """
    IRLS/Newton-Raphson on the REGULARIZED (MAP) objective: Gaussian prior
    N(0, alpha^-1 I) on the feature weights (bias left unregularized, as
    is conventional). Returns both w_MAP and the final Hessian H, which
    doubles as the precision matrix of the Laplace-approximated posterior.
    """
    X_bias = np.column_stack([np.ones(len(X)), X])
    n_features = X_bias.shape[1]
    reg_vec = np.concatenate([[0.0], np.full(n_features - 1, alpha)])  # no prior penalty on bias
    w = np.zeros(n_features)

    H = None
    for _ in range(n_iter):
        p = sigmoid(X_bias @ w)
        grad = X_bias.T @ (p - y) + reg_vec * w
        R = p * (1 - p)
        H = (X_bias * R[:, None]).T @ X_bias + np.diag(reg_vec)
        step = np.linalg.solve(H, grad)
        w = w - step
        if np.linalg.norm(step) < 1e-10:
            break

    return w, H


def laplace_predictive_proba(X_new: np.ndarray, w_map: np.ndarray, H: np.ndarray):
    """
    Bishop eq. 4.153: approximate P(C1|x) by integrating the sigmoid over
    the Laplace-approximated Gaussian posterior on the activation a = w.x.
    Returns (bayesian_proba, plugin_proba) so the two can be compared
    point by point.
    """
    X_bias = np.column_stack([np.ones(len(X_new)), X_new])
    Sigma = np.linalg.inv(H)

    mu_a = X_bias @ w_map
    sigma_a2 = np.einsum("ij,jk,ik->i", X_bias, Sigma, X_bias)
    kappa = 1.0 / np.sqrt(1 + np.pi * sigma_a2 / 8)

    bayesian_proba = sigmoid(kappa * mu_a)
    plugin_proba = sigmoid(mu_a)  # naive: ignores weight uncertainty entirely
    return bayesian_proba, plugin_proba, sigma_a2


def compare_plugin_vs_bayesian(splits, w_map, H):
    print("\nPlug-in MAP vs. Laplace-approximated Bayesian predictive probabilities:")
    for split_name in ["train", "val", "test"]:
        X, y = splits[f"X_{split_name}"], splits[f"y_{split_name}"]
        bayes_proba, plugin_proba, sigma_a2 = laplace_predictive_proba(X, w_map, H)

        for name, proba in [("plug-in MAP", plugin_proba), ("Bayesian (Laplace)", bayes_proba)]:
            preds = (proba >= 0.5).astype(int)
            acc = accuracy_score(y, preds)
            ll = log_loss(y, np.clip(proba, 1e-9, 1 - 1e-9))
            brier = brier_score_loss(y, proba)
            print(f"  [{split_name:<5}] {name:<19}: accuracy={acc:.4f}  log-loss={ll:.4f}  Brier={brier:.4f}")

        avg_shrink = np.mean(np.abs(plugin_proba - 0.5) - np.abs(bayes_proba - 0.5))
        print(f"  [{split_name:<5}] average confidence pulled toward 0.5 by: {avg_shrink:.4f}  "
              f"(average posterior variance of the activation: {sigma_a2.mean():.4f})")

    print(
        "\n  note: the correction here is tiny because there are thousands of "
        "training rows relative to only 12 features, so the posterior over w "
        "is already tight (low sigma_a^2) and plug-in MAP is a fine "
        "approximation. The toy demo above is the more honest place to SEE "
        "the effect: with only 40 points, posterior uncertainty is large and "
        "the Bayesian contours are visibly softer near/beyond the training "
        "cluster. The lesson generalizes: the fewer rows per parameter you "
        "have, the more the Laplace correction matters."
    )


def toy_2d_confidence_contours():
    """
    Small synthetic 2D dataset (so decision surface can be drawn), showing
    how plug-in MAP probability contours are much sharper/overconfident
    than the Laplace-Bayesian ones, especially far from the training
    cluster where the model has seen little data.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_per_class = 20
    class0 = rng.normal(loc=[-1, -1], scale=0.6, size=(n_per_class, 2))
    class1 = rng.normal(loc=[1, 1], scale=0.6, size=(n_per_class, 2))
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    w_map, H = map_logistic_regression(X, y, alpha=0.5)

    grid_x, grid_y = np.meshgrid(np.linspace(-5, 5, 150), np.linspace(-5, 5, 150))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    bayes_proba, plugin_proba, _ = laplace_predictive_proba(grid_points, w_map, H)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, proba, title in [
        (axes[0], plugin_proba, "Plug-in MAP: P(C1|x) = sigmoid(w_MAP . x)"),
        (axes[1], bayes_proba, "Laplace-Bayesian: uncertainty pulls P(C1|x) toward 0.5"),
    ]:
        contour = ax.contourf(grid_x, grid_y, proba.reshape(grid_x.shape), levels=20, cmap="RdBu_r", vmin=0, vmax=1)
        ax.scatter(class0[:, 0], class0[:, 1], color="black", marker="o", label="class 0")
        ax.scatter(class1[:, 0], class1[:, 1], color="black", marker="x", label="class 1")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.legend(fontsize=8, loc="lower right")
    fig.colorbar(contour, ax=axes, label="P(C1 | x)", fraction=0.046, pad=0.04)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "laplace_confidence_contours.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("Laplace approximation & Bayesian logistic regression (Unit 2)")

    print("\n-- Toy demo: plug-in vs. Bayesian confidence contours --")
    toy_2d_confidence_contours()

    print("\n-- Real stock data --")
    df = load_model_ready_dataset()
    splits = get_model_ready_splits(df, CLASSIFICATION_TARGET)
    print_split_sizes(splits, "Laplace / Bayesian logistic regression")

    w_map, H = map_logistic_regression(splits["X_train"], splits["y_train"])
    compare_plugin_vs_bayesian(splits, w_map, H)


if __name__ == "__main__":
    main()
