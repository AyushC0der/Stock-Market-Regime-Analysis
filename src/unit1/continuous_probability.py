"""
Unit 1 - Continuous random variables.

Syllabus topics covered here:
    - Continuous random variables
    - Probability densities
    - Quantiles
    - Mean and variance
    - Expectation

We treat AAPL's daily return (Return_1D) as a continuous random variable X
and walk through every core idea a syllabus would ask about it:

    - a probability DENSITY (not a probability mass) describes X, so
      P(X = exactly 0.01234...) is technically zero - only P(a <= X <= b)
      (an area under the density) means anything;
    - the empirical density can be estimated with a histogram/KDE and
      compared against a fitted parametric density (Normal, Student-t);
    - QUANTILES of that density are literally what "Value at Risk" is in
      finance - the q-quantile is the loss threshold breached (100-q)% of
      the time;
    - MEAN and VARIANCE are the first two moments of the distribution;
    - EXPECTATION E[X] is defined as an integral of x * f(x) over the
      density - we compute it two ways (the sample mean, and a numerical
      integral of the fitted density) and show they roughly agree.

Run: python -m src.unit1.continuous_probability   (or via run_unit1.py)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("results", "unit1", "probability")


def load_returns(ticker: str = "AAPL") -> np.ndarray:
    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python main.py` first.")
    df = pd.read_csv(path)
    return df["Return_1D"].dropna().to_numpy()


def moments_summary(x: np.ndarray, ticker: str):
    """Mean, variance, std, skew, kurtosis - the first few moments of the distribution."""
    mean = np.mean(x)
    variance = np.var(x, ddof=1)  # sample variance, Bessel-corrected
    std = np.std(x, ddof=1)
    skew = stats.skew(x)
    kurtosis = stats.kurtosis(x)  # excess kurtosis (Normal = 0)

    print(f"\n{ticker} Return_1D - mean/variance summary (n={len(x)}):")
    print(f"  mean (E[X])        = {mean:.6f}   ({mean * 100:.4f}% average daily return)")
    print(f"  variance (Var[X])  = {variance:.8f}")
    print(f"  std dev            = {std:.6f}")
    print(f"  skew               = {skew:.4f}  (0 = symmetric; real returns are often left-skewed)")
    print(f"  excess kurtosis    = {kurtosis:.4f}  (0 = Normal-like tails; >0 = fatter tails than Normal)")
    return {"mean": mean, "variance": variance, "std": std, "skew": skew, "kurtosis": kurtosis}


def fit_densities(x: np.ndarray):
    """
    Maximum likelihood fits of two candidate densities for daily returns:
      - Normal:     the "textbook" assumption
      - Student-t:  has fatter tails, usually a much better fit for real
                    returns (large moves happen more often than a Normal
                    would predict - this is the well-known "fat tails"
                    stylized fact of financial returns).
    """
    norm_params = stats.norm.fit(x)                 # (mu, sigma)
    t_params = stats.t.fit(x)                        # (dof, loc, scale)

    ll_norm = np.sum(stats.norm.logpdf(x, *norm_params))
    ll_t = np.sum(stats.t.logpdf(x, *t_params))
    print(f"\nMLE fit comparison (higher log-likelihood = better fit to this sample):")
    print(f"  Normal  : mu={norm_params[0]:.6f}, sigma={norm_params[1]:.6f}  -> log-likelihood = {ll_norm:.2f}")
    print(f"  Student-t: dof={t_params[0]:.2f}, loc={t_params[1]:.6f}, scale={t_params[2]:.6f} "
          f"-> log-likelihood = {ll_t:.2f}")
    if ll_t > ll_norm:
        print("  -> Student-t fits better, i.e. real returns have fatter tails than a Normal assumes.")
    return norm_params, t_params


def quantiles_and_var(x: np.ndarray, norm_params, t_params):
    """
    Quantiles of the empirical data vs. the fitted densities. The 5%
    quantile of the *loss* distribution (i.e. -X) is the daily "Value at
    Risk" at the 95% confidence level: "on 95% of days, you won't lose
    more than this."
    """
    quantile_levels = [0.01, 0.05, 0.50, 0.95, 0.99]
    empirical_q = np.quantile(x, quantile_levels)
    normal_q = stats.norm.ppf(quantile_levels, *norm_params)
    t_q = stats.t.ppf(quantile_levels, *t_params)

    print("\nQuantiles of daily return (empirical vs. fitted densities):")
    header = f"  {'level':>6} | {'empirical':>10} | {'Normal fit':>10} | {'Student-t fit':>13}"
    print(header)
    for level, e, n, t in zip(quantile_levels, empirical_q, normal_q, t_q):
        print(f"  {level:>6.0%} | {e:>10.4%} | {n:>10.4%} | {t:>13.4%}")

    var_95 = -empirical_q[1]  # 5% quantile of returns = -(95% VaR) of losses
    print(f"\n  Empirical 1-day 95% Value at Risk: a loss of {var_95:.2%} or worse "
          f"happens on roughly 5% of trading days.")
    return empirical_q


def expectation_two_ways(x: np.ndarray, t_params):
    """
    E[X] is *defined* as integral of x * f(x) dx over the density f. The
    sample mean is an ESTIMATOR of that quantity, not the definition - so
    here we compute E[X] both ways and confirm they land close together:
      1) the sample mean (what everyone actually uses in practice)
      2) numerically integrating x * f_t(x) using the fitted Student-t
         density, where f_t is the MLE-fit density from fit_densities()
    """
    sample_mean = np.mean(x)

    def integrand(val):
        return val * stats.t.pdf(val, *t_params)

    # Student-t has heavy tails; integrate over a wide-enough range that
    # the remaining tail mass is negligible.
    integral_value, _ = integrate.quad(integrand, -1.0, 1.0)

    print("\nExpectation E[X], computed two ways:")
    print(f"  1) sample mean                         = {sample_mean:.6f}")
    print(f"  2) numerical integral of x * f_t(x) dx  = {integral_value:.6f}")
    print(f"  difference                              = {abs(sample_mean - integral_value):.6f}")


def probability_density_examples(x: np.ndarray, norm_params):
    """
    Concretely show that a density gives you *area*, not point
    probability: P(-1% <= X <= 1%) is computed as a CDF difference, using
    the fitted Normal density, and cross-checked against the empirical
    fraction of days that actually fall in that band.
    """
    lower, upper = -0.01, 0.01
    p_model = stats.norm.cdf(upper, *norm_params) - stats.norm.cdf(lower, *norm_params)
    p_empirical = np.mean((x >= lower) & (x <= upper))

    print(f"\nP({lower:.0%} <= Return_1D <= {upper:.0%}):")
    print(f"  from fitted Normal density (CDF difference) = {p_model:.4f}")
    print(f"  empirical fraction of days in that band     = {p_empirical:.4f}")


def plot_density_comparison(x: np.ndarray, norm_params, t_params, ticker: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=80, density=True, alpha=0.4, color="gray", label="empirical histogram")

    grid = np.linspace(x.min(), x.max(), 500)
    ax.plot(grid, stats.norm.pdf(grid, *norm_params), label="Normal MLE fit", linewidth=2)
    ax.plot(grid, stats.t.pdf(grid, *t_params), label="Student-t MLE fit", linewidth=2)

    ax.set_xlabel("Return_1D")
    ax.set_ylabel("density")
    ax.set_title(f"{ticker} daily return: empirical density vs. fitted models")
    ax.legend()
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{ticker}_return_density.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main(ticker: str = "AAPL"):
    print(f"Continuous probability theory (Unit 1) - {ticker} daily returns")
    x = load_returns(ticker)

    moments_summary(x, ticker)
    norm_params, t_params = fit_densities(x)
    quantiles_and_var(x, norm_params, t_params)
    probability_density_examples(x, norm_params)
    expectation_two_ways(x, t_params)
    plot_density_comparison(x, norm_params, t_params, ticker)


if __name__ == "__main__":
    main()
