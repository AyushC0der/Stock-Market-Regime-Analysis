"""
Unit 2 runner: linear models for regression + classification, all applied
to the processed stock data from Milestone 1.

Regression (predicting Target_Return):
    1. Least squares / MLE            (normal equations == Gaussian MLE)
    2. Robust regression              (Huber loss vs. OLS under outliers)
    3. Ridge regression                (L2-regularized, shrinkage path)
    4. Bayesian linear regression      (posterior over weights, predictive uncertainty)

Classification (predicting Target_Direction):
    5. Generative model                (shared-covariance Gaussian discriminant)
    6. Discriminative model            (logistic regression via IRLS)
    7. Laplace approx / Bayesian logistic regression
    8. Kernels & SVMs                   (kernel trick, kernel ridge, SVM)

Requires data/processed/*_features.csv to already exist - run `python
main.py` first if you haven't.

Run: python run_unit2.py
Outputs: figures + console summaries under results/unit2/
"""

from src.unit2.regression import (
    least_squares, robust_regression, ridge_regression, bayesian_linear_regression,
)
from src.unit2.classification import (
    generative_models, discriminative_models, laplace_bayesian_logistic, kernels_svm,
)


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_unit2():
    section("1/8 - Least squares / MLE regression")
    least_squares.main()

    section("2/8 - Robust regression")
    robust_regression.main()

    section("3/8 - Ridge regression")
    ridge_regression.main()

    section("4/8 - Bayesian linear regression")
    bayesian_linear_regression.main()

    section("5/8 - Probabilistic generative classification model")
    generative_models.main()

    section("6/8 - Probabilistic discriminative classification model")
    discriminative_models.main()

    section("7/8 - Laplace approximation & Bayesian logistic regression")
    laplace_bayesian_logistic.main()

    section("8/8 - Kernels, the kernel trick, and SVMs")
    kernels_svm.main()

    section("Unit 2 complete")
    print("All figures and summaries were saved under results/unit2/.")
    print("The full Unit 1 + Unit 2 syllabus scope is now implemented.")


if __name__ == "__main__":
    run_unit2()
