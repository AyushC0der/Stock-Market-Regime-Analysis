"""
Unit 1 - Polynomial curve fitting & the bias-variance tradeoff.

Syllabus topics covered here:
    - What is machine learning, and why (supervised learning = "learn a
      function from labelled examples")
    - Polynomial curve fitting (the textbook "hello world" of ML)

We do this in two parts:

    PART A - the classic textbook demo (Bishop, PRML Ch.1): fit
    y = sin(2*pi*x) + noise with polynomials of increasing degree M,
    and watch training error fall monotonically while held-out error
    forms a U-shape. This is where "overfitting" and "underfitting"
    get their precise, non-hand-wavy meaning.

    PART B - the exact same idea, but on real AAPL price data instead
    of a synthetic sine wave, so you can see that a stock price chart
    over a short window is *also* just a curve you're fitting a
    polynomial to - and that a high-degree polynomial that hugs every
    wiggle in the training window is a textbook overfit that falls
    apart on the next few days.

Run: python -m src.unit1.polynomial_curve_fitting   (or via run_unit1.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("results", "unit1", "polynomial")
PROCESSED_DIR = os.path.join("data", "processed")

RNG_SEED = 42  # fixed seed -> reproducible noise/plots


# ---------------------------------------------------------------------------
# PART A: the classic synthetic sine-wave demo
# ---------------------------------------------------------------------------

def generate_synthetic_curve_data(n_points: int = 10, noise_std: float = 0.25):
    """
    The textbook example: draw n_points x-values in [0, 1], the "true"
    function is sin(2*pi*x), and we only ever observe it through Gaussian
    noise. This is supervised learning in miniature: we're given (x, y)
    pairs and want to recover a function that generalizes to new x.
    """
    rng = np.random.default_rng(RNG_SEED)
    x = np.sort(rng.uniform(0, 1, n_points))
    true_y = np.sin(2 * np.pi * x)
    y = true_y + rng.normal(0, noise_std, n_points)
    return x, y


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    Least-squares polynomial fit via the normal equations, i.e. exactly
    the "maximum likelihood under Gaussian noise = minimize sum of squared
    errors" result you get from Unit 2's MLE derivation - we just don't
    derive it yet, we use it. Returns coefficients highest-degree first
    (numpy convention), fit with the design matrix
        Phi = [1, x, x^2, ..., x^M]
    and w = (Phi^T Phi)^-1 Phi^T y.
    """
    return np.polyfit(x, y, degree)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def synthetic_overfitting_demo():
    """
    Fit degree 1, 3, 9 polynomials to 10 noisy points and plot against the
    true sin curve. Degree 9 has enough free parameters (10 coefficients)
    to pass through nearly every training point exactly - and that is
    precisely the overfitting failure mode: zero training error, wild
    behaviour anywhere else.
    """
    x_train, y_train = generate_synthetic_curve_data(n_points=10)
    x_dense = np.linspace(0, 1, 300)
    true_curve = np.sin(2 * np.pi * x_dense)

    degrees = [1, 3, 9]
    fig, axes = plt.subplots(1, len(degrees), figsize=(15, 4.5), sharey=True)
    for ax, degree in zip(axes, degrees):
        coeffs = fit_polynomial(x_train, y_train, degree)
        y_fit = np.polyval(coeffs, x_dense)
        train_rmse = rmse(y_train, np.polyval(coeffs, x_train))

        ax.plot(x_dense, true_curve, "g--", label="true sin(2pix)", linewidth=1.5)
        ax.scatter(x_train, y_train, color="tab:blue", label="noisy samples", zorder=5)
        ax.plot(x_dense, y_fit, color="tab:red", label=f"degree {degree} fit")
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(f"M = {degree}\ntrain RMSE = {train_rmse:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle("Underfitting vs. overfitting: same 10 points, different polynomial degree")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "synthetic_underfit_overfit.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def synthetic_train_vs_test_error_curve():
    """
    The real point of the demo: plot train error and *held-out* error as
    a function of polynomial degree M = 0..9. Train error falls
    monotonically (more parameters can only fit the training set better
    or equal). Held-out error falls, bottoms out, then rises again - that
    turning point is the sweet spot, and everything past it is overfitting.
    """
    x_train, y_train = generate_synthetic_curve_data(n_points=10)
    # An independent, unseen sample from the *same* true process - this is
    # the "held-out set" role that Unit 2/4 will formalize with train/val/test.
    x_test, y_test = generate_synthetic_curve_data(n_points=100)
    # regenerate test with a different, unseen slice of noise
    rng = np.random.default_rng(RNG_SEED + 1)
    x_test = np.sort(rng.uniform(0, 1, 100))
    y_test = np.sin(2 * np.pi * x_test) + rng.normal(0, 0.25, 100)

    degrees = list(range(0, 10))
    train_errors, test_errors = [], []
    for degree in degrees:
        coeffs = fit_polynomial(x_train, y_train, degree)
        train_errors.append(rmse(y_train, np.polyval(coeffs, x_train)))
        test_errors.append(rmse(y_test, np.polyval(coeffs, x_test)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(degrees, train_errors, "o-", label="training RMSE")
    ax.plot(degrees, test_errors, "o-", label="held-out RMSE")
    best_degree = degrees[int(np.argmin(test_errors))]
    ax.axvline(best_degree, color="gray", linestyle=":", label=f"best degree = {best_degree}")
    ax.set_xlabel("polynomial degree M")
    ax.set_ylabel("RMSE")
    ax.set_title("Training error keeps falling; held-out error forms a U")
    ax.legend()
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "synthetic_train_vs_test_error.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")
    return best_degree


# ---------------------------------------------------------------------------
# PART B: the same idea, applied to real stock prices
# ---------------------------------------------------------------------------

def load_price_series(ticker: str = "AAPL") -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python main.py` first to generate processed data."
        )
    return pd.read_csv(path, parse_dates=["Date"])


def stock_overfitting_demo(ticker: str = "AAPL", train_days: int = 30, test_days: int = 10):
    """
    Take one contiguous 40-trading-day slice of real price history: fit a
    polynomial of the day-index (0..29) to the adjusted close price using
    the first 30 days as "training data", then see how well each degree's
    curve extrapolates to the next 10 unseen days. This is exactly the
    synthetic demo above, except the "true function" is now a real,
    noisy stock price - and there IS no clean underlying curve, which is
    itself an honest lesson: financial time series are far noisier than
    the textbook sine wave, so overfitting bites even harder, even sooner.
    """
    df = load_price_series(ticker)
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    # Pick a slice roughly in the middle of the history so both a training
    # and a following test window exist comfortably inside the data.
    start = len(df) // 2
    window = df.iloc[start: start + train_days + test_days].reset_index(drop=True)
    if len(window) < train_days + test_days:
        raise ValueError("Not enough rows for the requested train/test window.")

    day_idx = np.arange(len(window))
    prices = window[price_col].to_numpy()

    x_train, y_train = day_idx[:train_days], prices[:train_days]
    x_test, y_test = day_idx[train_days:], prices[train_days:]

    # Scale the x-axis to [0, 1] purely for numerical conditioning of
    # np.polyfit at high degree - this does not change what's being taught.
    x_scale = train_days - 1
    x_train_s, x_test_s, day_idx_s = x_train / x_scale, x_test / x_scale, day_idx / x_scale

    degrees = list(range(1, 15))
    train_errors, test_errors = [], []
    for degree in degrees:
        coeffs = fit_polynomial(x_train_s, y_train, degree)
        train_errors.append(rmse(y_train, np.polyval(coeffs, x_train_s)))
        test_errors.append(rmse(y_test, np.polyval(coeffs, x_test_s)))

    best_degree = degrees[int(np.argmin(test_errors))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: a couple of example fits over the real price window
    ax1.plot(day_idx, prices, "o", color="black", markersize=4, label="actual price")
    ax1.axvline(train_days - 0.5, color="gray", linestyle="--", label="train/test boundary")
    for degree, style in [(1, "tab:blue"), (best_degree, "tab:green"), (degrees[-1], "tab:red")]:
        coeffs = fit_polynomial(x_train_s, y_train, degree)
        ax1.plot(day_idx, np.polyval(coeffs, day_idx_s), style, label=f"degree {degree} fit")
    ax1.set_xlabel("trading day index")
    ax1.set_ylabel(f"{ticker} {price_col}")
    ax1.set_title(f"{ticker}: fitting a curve to {train_days} days, testing on the next {test_days}")
    ax1.legend(fontsize=8)

    # Right: train vs test error vs degree, same shape as the synthetic plot
    ax2.plot(degrees, train_errors, "o-", label="training RMSE")
    ax2.plot(degrees, test_errors, "o-", label="held-out RMSE")
    ax2.axvline(best_degree, color="gray", linestyle=":", label=f"best degree = {best_degree}")
    ax2.set_xlabel("polynomial degree M")
    ax2.set_ylabel("RMSE ($)")
    ax2.set_title("Same overfitting pattern, real (noisier) data")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{ticker}_price_curve_fitting.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")
    return best_degree


def main():
    print("Polynomial curve fitting (Unit 1)")
    print("-- Part A: classic sin(2pix) textbook demo --")
    synthetic_overfitting_demo()
    best_synthetic_degree = synthetic_train_vs_test_error_curve()
    print(f"  best degree on synthetic data: {best_synthetic_degree}")

    print("-- Part B: same idea on real AAPL price data --")
    best_stock_degree = stock_overfitting_demo("AAPL")
    print(f"  best degree on AAPL price window: {best_stock_degree}")
    print(
        "  note: a low best-degree here is the expected, healthy result - "
        "it means the held-out error correctly punished high-degree "
        "polynomials for memorizing noise instead of a real trend."
    )


if __name__ == "__main__":
    main()
