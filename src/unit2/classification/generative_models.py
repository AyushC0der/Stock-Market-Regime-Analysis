"""
Unit 2 - Linear models for classification: discriminant functions via
probabilistic GENERATIVE models.

Syllabus topics covered here:
    - Discriminant function
    - Probabilistic generative models

A generative classifier models each class's data distribution directly -
P(x | C_k) - then uses Bayes' rule (Unit 1, again) to flip it around into
the posterior we actually want:

    P(C_k | x) = P(x | C_k) P(C_k) / P(x)

Here we assume each class's features are Gaussian, SHARING one covariance
matrix Sigma across both classes (only the means differ). Under that
assumption, the log posterior-odds is provably LINEAR in x:

    log [P(C1|x) / P(C0|x)] = w^T x + w0
    where  w  = Sigma^-1 (mu1 - mu0)
           w0 = -1/2 mu1^T Sigma^-1 mu1 + 1/2 mu0^T Sigma^-1 mu0 + log(pi1/pi0)

This IS a linear discriminant function - and it's exactly what "Linear
Discriminant Analysis" (LDA) computes, which is why we cross-check our
by-hand derivation against sklearn's LinearDiscriminantAnalysis below.

We predict Target_Direction (up=1 / down=0) from the engineered features.

Run: python -m src.unit2.classification.generative_models   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    CLASSIFICATION_TARGET,
)

RESULTS_DIR = os.path.join("results", "unit2", "classification")


def fit_shared_covariance_generative_model(X: np.ndarray, y: np.ndarray):
    """
    Estimate class priors, class-conditional means, and one pooled
    (shared) covariance matrix from labelled training data - the
    generative model's parameters, estimated by maximum likelihood.
    """
    classes = [0, 1]
    means, priors = {}, {}
    n_total = len(y)
    pooled_scatter = np.zeros((X.shape[1], X.shape[1]))

    for c in classes:
        X_c = X[y == c]
        means[c] = X_c.mean(axis=0)
        priors[c] = len(X_c) / n_total
        centered = X_c - means[c]
        pooled_scatter += centered.T @ centered

    # Shared covariance = pooled within-class scatter / (N - n_classes),
    # the standard unbiased estimate used by LDA.
    shared_cov = pooled_scatter / (n_total - len(classes))
    return means, priors, shared_cov


def discriminant_weights(means: dict, priors: dict, shared_cov: np.ndarray):
    """w and w0 for log P(C1|x)/P(C0|x) = w.x + w0, derived from the Gaussian
    generative model with shared covariance (Bishop 4.33-4.36)."""
    cov_inv = np.linalg.inv(shared_cov)
    w = cov_inv @ (means[1] - means[0])
    w0 = (
        -0.5 * means[1] @ cov_inv @ means[1]
        + 0.5 * means[0] @ cov_inv @ means[0]
        + np.log(priors[1] / priors[0])
    )
    return w, w0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba_generative(X, w, w0):
    return sigmoid(X @ w + w0)


def verify_against_sklearn_lda(X_train, y_train, w, w0):
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None)
    lda.fit(X_train, y_train)

    # Compare DECISION DIRECTION, not raw coefficients: LDA's internal
    # parametrization can differ by an overall scale factor while still
    # producing the identical decision boundary and posterior ranking.
    manual_direction = w / np.linalg.norm(w)
    sklearn_direction = lda.coef_[0] / np.linalg.norm(lda.coef_[0])
    cosine_similarity = np.dot(manual_direction, sklearn_direction)

    print("Verifying our by-hand generative-model discriminant against sklearn's LDA:")
    print(f"  cosine similarity between decision-boundary directions = {cosine_similarity:.6f} "
          f"(should be ~1.0 or ~-1.0)")
    return lda


def full_feature_evaluation(splits, w, w0):
    print("\nFull-feature generative model, predicting Target_Direction:")
    for split_name in ["train", "val", "test"]:
        X, y = splits[f"X_{split_name}"], splits[f"y_{split_name}"]
        proba = predict_proba_generative(X, w, w0)
        preds = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        ll = log_loss(y, np.clip(proba, 1e-9, 1 - 1e-9))
        print(f"  {split_name:<5}: accuracy={acc:.4f}  log-loss={ll:.4f}")
    print(
        "  note: accuracy in the low-to-mid 50s on test is the expected, honest "
        "result here - see the README for why that's a correct outcome, not a "
        "modeling failure."
    )


def plot_2d_decision_boundary(df, feature_a="RSI", feature_b="MACD"):
    """
    Refit the SAME generative model on just two features, purely so the
    linear decision boundary can be drawn on a 2D scatter plot - the
    full model above uses all features and can't be visualized this way.
    """
    X2 = df[[feature_a, feature_b]].to_numpy()
    y = df[CLASSIFICATION_TARGET].to_numpy()

    # Standardize these two features the same way the full pipeline does,
    # fit on the whole slice purely for the plot (this plot is illustrative,
    # not a leakage-sensitive evaluation - no metric is read off this fit).
    X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)

    means, priors, shared_cov = fit_shared_covariance_generative_model(X2, y)
    w, w0 = discriminant_weights(means, priors, shared_cov)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c, color, label in [(0, "tab:red", "down"), (1, "tab:blue", "up")]:
        subset = X2[y == c]
        ax.scatter(subset[:, 0], subset[:, 1], s=6, alpha=0.25, color=color, label=f"Target_Direction={c} ({label})")

    x_grid = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 200)
    # Decision boundary: w0 + w[0]*x + w[1]*y = 0  ->  y = -(w0 + w[0]*x) / w[1]
    y_grid = -(w0 + w[0] * x_grid) / w[1]
    ax.plot(x_grid, y_grid, color="black", linewidth=2, label="linear discriminant boundary")
    ax.set_xlim(X2[:, 0].min(), X2[:, 0].max())
    ax.set_ylim(X2[:, 1].min(), X2[:, 1].max())
    ax.set_xlabel(f"{feature_a} (standardized)")
    ax.set_ylabel(f"{feature_b} (standardized)")
    ax.set_title("Probabilistic generative model: shared-covariance Gaussian -> linear boundary")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "generative_model_decision_boundary.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("Probabilistic generative classification model (Unit 2)")
    df = load_model_ready_dataset()
    splits = get_model_ready_splits(df, CLASSIFICATION_TARGET)
    print_split_sizes(splits, "Generative model")

    means, priors, shared_cov = fit_shared_covariance_generative_model(splits["X_train"], splits["y_train"])
    w, w0 = discriminant_weights(means, priors, shared_cov)
    verify_against_sklearn_lda(splits["X_train"], splits["y_train"], w, w0)
    full_feature_evaluation(splits, w, w0)
    plot_2d_decision_boundary(df)


if __name__ == "__main__":
    main()
