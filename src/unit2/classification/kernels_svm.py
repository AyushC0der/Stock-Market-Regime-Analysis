"""
Unit 2 - Kernel functions, using kernels in GLMs, the kernel trick, and SVMs.

Syllabus topics covered here:
    - Kernel functions
    - Using kernels in GLMs
    - Kernel trick
    - SVMs

A kernel k(x, x') is a shortcut for computing an inner product in some
(possibly very high- or infinite-dimensional) feature space phi(x), WITHOUT
ever forming phi(x) explicitly:

    k(x, x') = phi(x) . phi(x')

The "kernel trick" is: if a model's math only ever needs x through inner
products x_i . x_j (which is true of ridge regression's dual form, and of
the SVM's dual form), you can swap every such inner product for k(x_i,
x_j) and the model implicitly operates in the richer feature space phi
for free - no matter how big phi's dimension is.

PART A demonstrates the trick concretely: we build an explicit degree-2
polynomial feature map for a small feature set, compute inner products
the "slow way" (form phi(x) for every point, then dot them), and confirm
that's numerically identical to the polynomial KERNEL FUNCTION applied
directly to the raw x's - i.e. the kernel really is just a fast way to
get the same numbers.

PART B is "using kernels in a GLM": kernel ridge regression is exactly
ridge regression (Unit 2) with every dot product replaced by a kernel,
predicting Target_Return.

PART C is the SVM: a linear-kernel SVM is (almost) a max-margin version
of logistic regression's decision boundary; an RBF-kernel SVM can bend
that boundary into non-linear shapes. We compare both, predicting
Target_Direction.

Run: python -m src.unit2.classification.kernels_svm   (or via run_unit2.py)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error

from src.unit2.data_utils import (
    load_model_ready_dataset, get_model_ready_splits, print_split_sizes,
    REGRESSION_TARGET, CLASSIFICATION_TARGET, RANDOM_SEED,
)

RESULTS_DIR = os.path.join("results", "unit2", "classification")

# Kernel methods scale ~O(n^2)-O(n^3); subsample the training set purely
# for tractability. Test/val sets are left full-size for honest evaluation.
KERNEL_TRAIN_SAMPLE = 1500


def subsample_train(splits, n=KERNEL_TRAIN_SAMPLE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(splits["y_train"]), size=min(n, len(splits["y_train"])), replace=False)
    return splits["X_train"][idx], splits["y_train"][idx]


# ---------------------------------------------------------------------------
# PART A: the kernel trick, made concrete
# ---------------------------------------------------------------------------

def explicit_degree2_feature_map(X: np.ndarray) -> np.ndarray:
    """
    Explicit phi(x) for a degree-2 polynomial kernel (x.x' + 1)^2 over a
    small number of raw dimensions d: includes 1, all x_i, all x_i*x_j
    (i<=j) - exactly the monomials that expanding (x.x'+1)^2 produces.
    We only do this for a handful of features since the explicit feature
    space grows quadratically - the entire point of the kernel trick is
    to avoid ever having to build this for large feature sets.
    """
    n, d = X.shape
    features = [np.ones(n)]
    for i in range(d):
        features.append(np.sqrt(2) * X[:, i])
    for i in range(d):
        for j in range(i, d):
            coef = 2 if i != j else 1
            features.append(np.sqrt(coef) * X[:, i] * X[:, j])
    return np.column_stack(features)


def verify_kernel_trick(X_small: np.ndarray):
    Phi = explicit_degree2_feature_map(X_small)
    K_explicit = Phi @ Phi.T
    K_kernel_function = polynomial_kernel(X_small, degree=2, gamma=1, coef0=1)

    max_diff = np.abs(K_explicit - K_kernel_function).max()
    print("Verifying the kernel trick: explicit phi(x).phi(x') vs. the polynomial kernel function")
    print(f"  feature-space dimension actually used: {Phi.shape[1]} "
          f"(from only {X_small.shape[1]} raw features)")
    print(f"  max |K_explicit - K_kernel(x,x')| = {max_diff:.10f}  (should be ~0)")
    print(
        "  -> same numbers, but the kernel function never had to build or "
        "store that expanded feature matrix - this is the entire trick, and "
        "it matters far more once the feature-space dimension explodes "
        "(e.g. an RBF kernel corresponds to an INFINITE-dimensional phi)."
    )


# ---------------------------------------------------------------------------
# PART B: using kernels in a GLM - kernel ridge regression
# ---------------------------------------------------------------------------

def kernel_ridge_vs_linear_ridge(splits):
    X_train, y_train = subsample_train(splits)
    print(f"\nKernel ridge regression (RBF kernel) vs. plain ridge regression, "
          f"predicting Target_Return (n_train={len(y_train)} subsampled for tractability):")

    linear_ridge = Ridge(alpha=100.0).fit(X_train, y_train)
    rbf_ridge = KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1).fit(X_train, y_train)

    for name, model in [("linear ridge", linear_ridge), ("RBF kernel ridge", rbf_ridge)]:
        preds = model.predict(splits["X_test"])
        rmse = np.sqrt(mean_squared_error(splits["y_test"], preds))
        print(f"  {name:<17}: test RMSE = {rmse:.6f}")
    print(
        "  note: an RBF kernel model is far more flexible (effectively infinite-"
        "dimensional feature space) - if it doesn't clearly beat the simple "
        "linear model on held-out data here, that's further evidence the "
        "signal in next-day returns is genuinely weak, not that the kernel "
        "model was mis-specified."
    )


# ---------------------------------------------------------------------------
# PART C: SVMs
# ---------------------------------------------------------------------------

def svm_full_feature_comparison(splits):
    X_train, y_train = subsample_train(splits)
    print(f"\nSVM classification, predicting Target_Direction (n_train={len(y_train)} subsampled):")

    for kernel in ["linear", "rbf"]:
        svm = SVC(kernel=kernel, C=1.0, gamma="scale")
        svm.fit(X_train, y_train)
        test_acc = accuracy_score(splits["y_test"], svm.predict(splits["X_test"]))
        print(f"  kernel={kernel:<6}: test accuracy = {test_acc:.4f}, "
              f"support vectors used = {svm.n_support_.sum()} / {len(y_train)}")


def svm_2d_decision_boundary(df, feature_a="RSI", feature_b="MACD"):
    """Visualize how the linear vs. RBF kernel bends the SVM decision boundary on 2 features."""
    X2 = df[[feature_a, feature_b]].to_numpy()
    y = df[CLASSIFICATION_TARGET].to_numpy()
    X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X2), size=min(1000, len(X2)), replace=False)
    X2, y = X2[idx], y[idx]

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, kernel in zip(axes, ["linear", "rbf"]):
        svm = SVC(kernel=kernel, C=1.0, gamma="scale").fit(X2, y)
        preds = svm.predict(grid_points).reshape(grid_x.shape)
        ax.contourf(grid_x, grid_y, preds, levels=[-0.5, 0.5, 1.5], alpha=0.3, colors=["tab:red", "tab:blue"])
        for c, color, label in [(0, "tab:red", "down"), (1, "tab:blue", "up")]:
            subset = X2[y == c]
            ax.scatter(subset[:, 0], subset[:, 1], s=8, alpha=0.5, color=color, label=label)
        ax.set_title(f"SVM, kernel = {kernel}")
        ax.set_xlabel(f"{feature_a} (standardized)")
        ax.set_ylabel(f"{feature_b} (standardized)")
        ax.legend(fontsize=8)

    fig.suptitle("Kernel choice changes the SHAPE of the decision boundary an SVM can draw")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "svm_kernel_decision_boundaries.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("Kernel functions, kernels in GLMs, the kernel trick, and SVMs (Unit 2)")

    df = load_model_ready_dataset()

    print("\n-- Part A: the kernel trick --")
    small_sample = df[["RSI", "MACD", "Volatility_5D"]].to_numpy()[:200]
    small_sample = (small_sample - small_sample.mean(axis=0)) / small_sample.std(axis=0)
    verify_kernel_trick(small_sample)

    print("\n-- Part B: kernels in a GLM (kernel ridge regression) --")
    reg_splits = get_model_ready_splits(df, REGRESSION_TARGET)
    print_split_sizes(reg_splits, "Kernel ridge regression")
    kernel_ridge_vs_linear_ridge(reg_splits)

    print("\n-- Part C: SVMs --")
    clf_splits = get_model_ready_splits(df, CLASSIFICATION_TARGET)
    print_split_sizes(clf_splits, "SVM classification")
    svm_full_feature_comparison(clf_splits)
    svm_2d_decision_boundary(df)


if __name__ == "__main__":
    main()
