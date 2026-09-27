"""
Unit 1 - Expectation and covariance (multivariate).

Syllabus topics covered here:
    - Expectation and covariance (extended to multiple random variables)

Here each ticker's daily return is one random variable, and we have five
of them observed jointly on the same dates. That's exactly the setting
where covariance matters: not just "how much does AAPL's return vary" but
"how much does AAPL's return move together with NVDA's".

Core identities demonstrated numerically:
    Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
    For a portfolio return P = sum_i w_i * R_i:
        E[P]   = sum_i w_i * E[R_i]                         (linearity of expectation)
        Var[P] = w^T Cov(R) w                                (quadratic form)
    both derived from first principles and cross-checked against a Monte
    Carlo-style direct calculation on the historical sample.

Run: python -m src.unit1.covariance_expectation   (or via run_unit1.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("results", "unit1", "probability")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def load_return_matrix() -> pd.DataFrame:
    """
    Build one wide DataFrame: rows = dates, columns = tickers, values =
    Return_1D, keeping only dates where ALL tickers have a value (an
    inner join on Date) so covariance is computed on a strictly aligned,
    same-day sample - misaligned dates would silently corrupt Cov(X, Y).
    """
    series = {}
    for ticker in TICKERS:
        path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run `python main.py` first.")
        df = pd.read_csv(path, parse_dates=["Date"])
        series[ticker] = df.set_index("Date")["Return_1D"]

    wide = pd.DataFrame(series).dropna(how="any")
    return wide


def covariance_and_correlation(returns: pd.DataFrame):
    cov = returns.cov()
    corr = returns.corr()

    print(f"\nCovariance matrix (n={len(returns)} aligned trading days):")
    print(cov.round(6))
    print("\nCorrelation matrix (covariance normalized to [-1, 1]):")
    print(corr.round(3))

    # Sanity checks that should always hold for a real covariance matrix.
    assert np.allclose(cov.values, cov.values.T), "Covariance matrix must be symmetric."
    eigenvalues = np.linalg.eigvalsh(cov.values)
    print(f"\n  smallest eigenvalue = {eigenvalues.min():.8f} "
          f"(should be >= 0 - covariance matrices are positive semi-definite)")

    return cov, corr


def portfolio_expectation_and_variance(returns: pd.DataFrame, cov: pd.DataFrame):
    """
    Equal-weight portfolio across the 5 tickers. Compute E[P] and Var[P]
    two ways: (a) the closed-form identities from linear combinations of
    random variables, and (b) directly from the realized daily portfolio
    return series - they should match almost exactly since (a) is just
    the algebra behind (b).
    """
    weights = np.full(len(TICKERS), 1 / len(TICKERS))
    means = returns.mean().to_numpy()

    # (a) closed form
    expected_return_formula = weights @ means
    variance_formula = weights @ cov.to_numpy() @ weights

    # (b) direct: actually build the portfolio return series and measure it
    portfolio_series = returns.to_numpy() @ weights
    expected_return_direct = portfolio_series.mean()
    variance_direct = portfolio_series.var(ddof=1)

    print("\nEqual-weight portfolio (20% each of AAPL/MSFT/GOOGL/AMZN/NVDA):")
    print(f"  E[P]   via  w * E[R]        = {expected_return_formula:.6f}")
    print(f"  E[P]   via  direct sample   = {expected_return_direct:.6f}")
    print(f"  Var[P] via  w^T Cov(R) w    = {variance_formula:.8f}")
    print(f"  Var[P] via  direct sample   = {variance_direct:.8f}")
    print(f"  Portfolio daily volatility (std) = {np.sqrt(variance_formula):.4%}")

    naive_avg_variance = np.mean(np.diag(cov.to_numpy()))
    print(f"\n  Note: average single-stock variance = {naive_avg_variance:.8f}, "
          f"but the portfolio's variance is lower ({variance_formula:.8f}) "
          f"whenever returns aren't perfectly correlated - this is diversification, "
          f"and it falls directly out of the Var[P] = w^T Cov(R) w identity.")


def plot_correlation_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Daily return correlation matrix")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "return_correlation_heatmap.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_pairwise_scatter(returns: pd.DataFrame, ticker_a: str = "AAPL", ticker_b: str = "NVDA"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(returns[ticker_a], returns[ticker_b], alpha=0.4, s=12)
    ax.set_xlabel(f"{ticker_a} Return_1D")
    ax.set_ylabel(f"{ticker_b} Return_1D")
    corr_val = returns[[ticker_a, ticker_b]].corr().iloc[0, 1]
    ax.set_title(f"{ticker_a} vs {ticker_b} daily returns (corr = {corr_val:.2f})")
    fig.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{ticker_a}_vs_{ticker_b}_scatter.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("Expectation & covariance across tickers (Unit 1)")
    returns = load_return_matrix()
    cov, corr = covariance_and_correlation(returns)
    portfolio_expectation_and_variance(returns, cov)
    plot_correlation_heatmap(corr)
    plot_pairwise_scatter(returns, "AAPL", "NVDA")


if __name__ == "__main__":
    main()
