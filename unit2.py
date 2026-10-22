"""Unit 2: regression, probabilistic classification, Bayes, kernels, and SVM."""

import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, BayesianRidge, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_squared_error,
                             precision_score, recall_score, r2_score, roc_auc_score)
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ["Return_1D", "Return_5D", "Return_20D", "Volatility_5D", "Volatility_20D",
            "Volume_Change", "RSI", "MACD", "Price_to_MA5", "Price_to_MA20",
            "Price_to_MA50", "Volume_to_MA"]


def heading(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def prepare(data):
    d = data.copy()
    price = d["Adj Close"] if "Adj Close" in d else d["Close"]
    for window in (5, 20, 50):
        d[f"Price_to_MA{window}"] = price / d[f"MA_{window}"] - 1
    d["Volume_to_MA"] = d["Volume"] / d["Volume_MA"] - 1
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["Target_Return", "Target_Direction"])
    train = d[d["Date"] <= "2021-12-31"]
    val = d[(d["Date"] > "2021-12-31") & (d["Date"] <= "2023-12-31")]
    test = d[d["Date"] > "2023-12-31"]
    scaler = StandardScaler().fit(train[FEATURES])
    x = {name: scaler.transform(split[FEATURES]) for name, split in
         [("train", train), ("val", val), ("test", test)]}
    y = {name: split["Target_Return"].to_numpy() for name, split in
         [("train", train), ("val", val), ("test", test)]}
    c = {name: split["Target_Direction"].to_numpy() for name, split in
         [("train", train), ("val", val), ("test", test)]}
    print(f"Chronological split: train={len(train)}, validation={len(val)}, test={len(test)} rows.")
    print("The scaler learns only from training dates; later data stays unseen until evaluation.")
    return d, x, y, c


def regression_lesson(x, y):
    heading("1. Linear models for regression: predict tomorrow's return")
    print("Regression predicts a number. We compare predictions with Target_Return.")
    print("RMSE is a typical error size (lower is better); R-squared compares with predicting the training-set mean.")
    train_x, val_x, test_x = x["train"], x["val"], x["test"]
    train_y, val_y, test_y = y["train"], y["val"], y["test"]

    ols = LinearRegression().fit(train_x, train_y)
    prediction = ols.predict(test_x)
    design = np.column_stack([np.ones(len(train_x)), train_x])
    manual_mle = np.linalg.lstsq(design, train_y, rcond=None)[0]
    sklearn_weights = np.r_[ols.intercept_, ols.coef_]
    print("\nMaximum likelihood and least squares:")
    print("Assuming Normal errors, maximizing likelihood gives the least-squares weights.")
    print(f"Manual normal-equation / sklearn coefficient difference={np.max(np.abs(manual_mle-sklearn_weights)):.8f}")
    print(f"OLS test: RMSE={np.sqrt(mean_squared_error(test_y, prediction)):.5f}, R2={r2_score(test_y, prediction):.4f}")

    robust = HuberRegressor(max_iter=500).fit(train_x, train_y)
    robust_pred = robust.predict(test_x)
    print("\nRobust regression (Huber): reduces the influence of unusually large errors.")
    print(f"Huber test RMSE={np.sqrt(mean_squared_error(test_y, robust_pred)):.5f}")
    rng = np.random.default_rng(42)
    contaminated_y = train_y.copy()
    outlier_rows = rng.choice(len(train_y), max(1, int(0.02 * len(train_y))), replace=False)
    contaminated_y[outlier_rows] += rng.choice([-0.15, 0.15], size=len(outlier_rows))
    dirty_ols = LinearRegression().fit(train_x, contaminated_y)
    dirty_huber = HuberRegressor(max_iter=500).fit(train_x, contaminated_y)
    ols_drift = np.linalg.norm(np.r_[dirty_ols.intercept_, dirty_ols.coef_] - sklearn_weights)
    huber_drift = np.linalg.norm(np.r_[dirty_huber.intercept_, dirty_huber.coef_] -
                                 np.r_[robust.intercept_, robust.coef_])
    print(f"With 2% deliberately added outliers, coefficient drift: OLS={ols_drift:.5f}, Huber={huber_drift:.5f}.")

    candidates = [0.1, 1, 10, 100, 1000, 100000]
    ridge_models = [Ridge(alpha=a).fit(train_x, train_y) for a in candidates]
    best = min(zip(candidates, ridge_models), key=lambda pair: mean_squared_error(val_y, pair[1].predict(val_x)))
    ridge_alpha, ridge = best
    ridge_pred = ridge.predict(test_x)
    print("\nRidge regression: penalizes large weights; choose penalty on validation dates.")
    print(f"Selected penalty={ridge_alpha}; test RMSE={np.sqrt(mean_squared_error(test_y, ridge_pred)):.5f}")

    bayes = BayesianRidge().fit(train_x, train_y)
    bayes_mean, bayes_std = bayes.predict(test_x, return_std=True)
    coverage = np.mean((test_y >= bayes_mean - 1.96 * bayes_std) & (test_y <= bayes_mean + 1.96 * bayes_std))
    print("\nBayesian linear regression: estimates weights probabilistically and reports prediction uncertainty.")
    print(f"Test RMSE={np.sqrt(mean_squared_error(test_y, bayes_mean)):.5f}; 95% interval coverage={coverage*100:.1f}%")

    sample = np.random.default_rng(42).choice(len(train_y), min(1200, len(train_y)), replace=False)
    kernel_model = KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1).fit(train_x[sample], train_y[sample])
    kernel_pred = kernel_model.predict(test_x)
    print("\nKernels in a linear model (kernel ridge): a kernel lets regression represent nonlinear patterns.")
    print(f"RBF kernel-ridge test RMSE={np.sqrt(mean_squared_error(test_y, kernel_pred)):.5f} "
          f"(fit on {len(sample)} training rows for speed)")


def explicit_degree2_map(x):
    """Explicit feature map whose dot product equals (1 + x dot z)^2."""
    columns = [np.ones(len(x))]
    columns.extend([np.sqrt(2) * x[:, i] for i in range(x.shape[1])])
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            columns.append((np.sqrt(2) if i != j else 1) * x[:, i] * x[:, j])
    return np.column_stack(columns)


def laplace_probabilities(model, x_train, x_eval):
    """Simple Laplace approximation around logistic-regression weights."""
    design = np.column_stack([np.ones(len(x_train)), x_train])
    weights = np.r_[model.intercept_[0], model.coef_[0]]
    p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
    hessian = (design * (p * (1 - p))[:, None]).T @ design
    hessian[1:, 1:] += np.eye(x_train.shape[1]) / model.C
    covariance = np.linalg.pinv(hessian)
    eval_design = np.column_stack([np.ones(len(x_eval)), x_eval])
    mean_activation = eval_design @ weights
    variance = np.maximum(np.einsum("ij,jk,ik->i", eval_design, covariance, eval_design), 0)
    shrink = 1 / np.sqrt(1 + np.pi * variance / 8)
    probability = 1 / (1 + np.exp(-np.clip(mean_activation * shrink, -30, 30)))
    return probability


def classification_lesson(x, c):
    heading("2. Linear models for classification: predict up or down")
    print("Classification predicts a category: 1 means up, 0 means down or zero return.")
    print("A confusion matrix counts correct and incorrect predictions for both classes.")
    train_x, test_x = x["train"], x["test"]
    train_y, test_y = c["train"], c["test"]

    lda = LinearDiscriminantAnalysis().fit(train_x, train_y)
    lda_pred = lda.predict(test_x)
    print("\nGenerative model (LDA): assumes Gaussian features with shared class covariance, then applies Bayes' rule.")
    print(f"Test accuracy={accuracy_score(test_y, lda_pred):.4f}")
    print("Its linear discriminant has form w^T x + b; the score determines which class is more likely.")

    logistic = LogisticRegression(C=1.0, max_iter=1000).fit(train_x, train_y)
    probability = logistic.predict_proba(test_x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    print("\nDiscriminative model (logistic regression): directly estimates P(up | features).")
    print(f"Test accuracy={accuracy_score(test_y, prediction):.4f}; precision={precision_score(test_y, prediction, zero_division=0):.4f}; "
          f"recall={recall_score(test_y, prediction, zero_division=0):.4f}; ROC-AUC={roc_auc_score(test_y, probability):.4f}")
    print("Confusion matrix (rows=actual [down, up], columns=predicted [down, up]):")
    print(confusion_matrix(test_y, prediction, labels=[0, 1]))

    bayes_prob = laplace_probabilities(logistic, train_x, test_x)
    print("\nBayesian logistic regression: Laplace approximation treats the fitted weights as uncertain.")
    print(f"Average probability adjustment toward 0.5 = {np.mean(np.abs(probability - bayes_prob)):.5f}")

    # Verify polynomial kernel equals dot products after an explicit feature expansion.
    demo = train_x[:30, :3]
    phi = explicit_degree2_map(demo)
    explicit = phi @ phi.T
    kernel = polynomial_kernel(demo, degree=2, gamma=1, coef0=1)
    print("\nKernel trick: (1 + x dot z)^2 matches the dot product in an expanded feature space.")
    print(f"Largest difference between both calculations = {np.max(np.abs(explicit-kernel)):.8f}")

    # Kernelized logistic regression: each training example is represented by
    # its similarities to a subset of training examples, computed by the RBF kernel.
    rng = np.random.default_rng(42)
    kernel_rows = rng.choice(len(train_y), min(1000, len(train_y)), replace=False)
    support_x, support_y = train_x[kernel_rows], train_y[kernel_rows]
    gamma = 1 / train_x.shape[1]
    train_kernel = rbf_kernel(support_x, support_x, gamma=gamma)
    test_kernel = rbf_kernel(test_x, support_x, gamma=gamma)
    kernel_glm = LogisticRegression(C=1.0, max_iter=1000).fit(train_kernel, support_y)
    print("Using RBF kernel similarities in a logistic GLM (kernel logistic regression): "
          f"test accuracy={accuracy_score(test_y, kernel_glm.predict(test_kernel)):.4f} "
          f"(fit on {len(kernel_rows)} training rows for speed)")
    print("The kernel provides nonlinear relationships as GLM inputs without hand-building a large feature map.")

    # SVM fit on a fixed sample for manageable runtime; test set remains chronological and untouched.
    take = rng.choice(len(train_y), min(1500, len(train_y)), replace=False)
    print("\nSupport Vector Machines find a wide-margin boundary; kernels allow curved boundaries.")
    print(f"For runtime, each SVM fits a random sample of {len(take)} training rows; test uses all {len(test_y)} rows.")
    for kind in ("linear", "rbf"):
        svm = SVC(kernel=kind, C=1.0, gamma="scale").fit(train_x[take], train_y[take])
        print(f"{kind.upper()} SVM test accuracy={accuracy_score(test_y, svm.predict(test_x)):.4f}")


def run(data):
    print("\nUNIT 2 - LINEAR MODELS")
    _, x, y, c = prepare(data)
    regression_lesson(x, y)
    classification_lesson(x, c)
    print("\nUnit 2 complete. Regression predicts a number; classification predicts a class.")
