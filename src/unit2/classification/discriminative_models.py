"""
Unit 2 - Linear models for classification: probabilistic DISCRIMINATIVE
models (logistic regression).

Syllabus topics covered here:
    - Probabilistic discriminative models

A discriminative classifier skips modeling P(x | C_k) entirely and
directly models the posterior P(C_k | x) as a parametric function of x -
here, the logistic sigmoid of a linear function:

    P(C1 | x) = sigmoid(w^T x)

Fitting w by maximum likelihood has NO closed form (unlike least squares),
because the sigmoid makes the log-likelihood nonlinear in w. Instead we
use Iteratively Reweighted Least Squares (IRLS) - Newton-Raphson applied
to the logistic log-likelihood - which repeatedly solves a WEIGHTED least
squares problem until w converges. We implement IRLS by hand and verify
it lands on the same w sklearn's LogisticRegression finds.

We then compare this discriminative model against last module's
GENERATIVE model on the identical train/val/test split: a classic result
in the field is that discriminative models tend to do at least as well,
often better, once the generative model's Gaussian-with-shared-covariance
assumption doesn't hold exactly (as here, since return-based features are
fat-tailed, not Gaussian - see Unit 1's continuous_probability module).

Run: python -m src.unit2.classification.discriminative_models   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, log_loss,
)

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    CLASSIFICATION_TARGET,
)
from src.unit2.classification.generative_models import (
    fit_shared_covariance_generative_model, discriminant_weights, predict_proba_generative,
)

RESULTS_DIR = os.path.join("results", "unit2", "classification")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def irls_logistic_regression(X: np.ndarray, y: np.ndarray, n_iter: int = 25, ridge: float = 1e-6):
    """
    Newton-Raphson / IRLS for logistic regression MLE:
        grad    = X^T (p - y)
        H       = X^T R X,   R = diag(p_i (1 - p_i))
        w_new   = w - H^-1 grad
    A tiny ridge term is added to H purely for numerical stability (in
    case R gets close to singular when predictions saturate near 0/1) -
    it is not meant as regularization in the ridge-regression sense.
    """
    X_bias = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X_bias.shape[1])

    for _ in range(n_iter):
        z = X_bias @ w
        p = sigmoid(z)
        grad = X_bias.T @ (p - y)
        R = p * (1 - p)
        H = (X_bias * R[:, None]).T @ X_bias + ridge * np.eye(X_bias.shape[1])
        step = np.linalg.solve(H, grad)
        w = w - step
        if np.linalg.norm(step) < 1e-10:
            break

    return w


def verify_irls_against_sklearn(X_train, y_train, w_irls):
    # C=np.inf -> unregularized MLE, matching our IRLS objective exactly.
    sk_model = LogisticRegression(C=np.inf, max_iter=1000)
    sk_model.fit(X_train, y_train)
    w_sklearn = np.concatenate([[sk_model.intercept_[0]], sk_model.coef_[0]])

    max_diff = np.abs(w_irls - w_sklearn).max()
    print("Verifying hand-written IRLS matches sklearn's (unregularized) LogisticRegression:")
    print(f"  max |w_irls - w_sklearn| = {max_diff:.6f}  (should be close to 0)")
    return sk_model


def evaluate_classifier(y_true, proba, label: str):
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, proba)
    ll = log_loss(y_true, np.clip(proba, 1e-9, 1 - 1e-9))
    print(f"  {label:<5}: accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  "
          f"f1={f1:.4f}  ROC-AUC={auc:.4f}  log-loss={ll:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "log_loss": ll}


def compare_generative_vs_discriminative(splits):
    """
    Fit both models on the identical training split, evaluate both on the
    identical test split - an apples-to-apples generative-vs-discriminative
    comparison.
    """
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    w_irls = irls_logistic_regression(X_train, y_train)
    X_test_bias = np.column_stack([np.ones(len(X_test)), X_test])
    proba_discriminative = sigmoid(X_test_bias @ w_irls)

    means, priors, shared_cov = fit_shared_covariance_generative_model(X_train, y_train)
    w_gen, w0_gen = discriminant_weights(means, priors, shared_cov)
    proba_generative = predict_proba_generative(X_test, w_gen, w0_gen)

    print("\nGenerative vs. discriminative, evaluated on the SAME held-out test set:")
    print(" Discriminative (logistic regression):")
    evaluate_classifier(y_test, proba_discriminative, "test")
    print(" Generative (shared-covariance Gaussian):")
    evaluate_classifier(y_test, proba_generative, "test")

    return w_irls, proba_discriminative, proba_generative


def plot_roc_curves(y_test, proba_discriminative, proba_generative):
    fig, ax = plt.subplots(figsize=(6, 6))
    for proba, label in [(proba_discriminative, "logistic regression"), (proba_generative, "generative model")]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random guessing (AUC=0.5)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve: generative vs. discriminative classifier (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "roc_curve_comparison.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("Probabilistic discriminative classification model (Unit 2)")
    df = load_model_ready_dataset()
    splits = get_model_ready_splits(df, CLASSIFICATION_TARGET)
    print_split_sizes(splits, "Discriminative model")

    w_irls = irls_logistic_regression(splits["X_train"], splits["y_train"])
    verify_irls_against_sklearn(splits["X_train"], splits["y_train"], w_irls)

    print("\nLogistic regression (IRLS), all three splits:")
    for split_name in ["train", "val", "test"]:
        X, y = splits[f"X_{split_name}"], splits[f"y_{split_name}"]
        X_bias = np.column_stack([np.ones(len(X)), X])
        proba = sigmoid(X_bias @ w_irls)
        evaluate_classifier(y, proba, split_name)

    _, proba_discriminative, proba_generative = compare_generative_vs_discriminative(splits)
    plot_roc_curves(splits["y_test"], proba_discriminative, proba_generative)


if __name__ == "__main__":
    main()
