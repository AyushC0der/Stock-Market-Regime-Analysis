"""Unit 1: ML introduction, curve fitting, and probability, in four lessons."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

OUT = "results/unit1_summary.png"


def heading(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def machine_learning_intro(data):
    heading("1. What is machine learning, and why use it?")
    print("Machine learning finds patterns in examples to make predictions or group data.")
    print("Supervised learning uses known answers: predict Target_Return or Target_Direction.")
    print("Unsupervised learning has no answer labels: group days with similar market behaviour.")
    print("Why use it? It can summarize many indicators and reveal patterns that are hard to spot by eye.")

    # K-means is an unsupervised example: it groups observations without using tomorrow's target.
    x = data[["Return_1D", "Volatility_20D"]].dropna().to_numpy()
    x = StandardScaler().fit_transform(x)
    labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(x)
    sizes = np.bincount(labels)
    print(f"Unsupervised example (K-means on today's return and volatility): cluster sizes {sizes.tolist()}")
    print("These clusters describe similar historical days; they are not up/down predictions.")


def polynomial_lesson():
    heading("2. Supervised learning and polynomial curve fitting")
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, 1, 30))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    degrees = [1, 3, 9]
    grid = np.linspace(0, 1, 250)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    train_errors, test_errors = [], []
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_train[:, None], y_train)
        train_errors.append(np.mean((model.predict(x_train[:, None]) - y_train) ** 2))
        test_errors.append(np.mean((model.predict(x_test[:, None]) - y_test) ** 2))
        axes[0].plot(grid, model.predict(grid[:, None]), label=f"degree {degree}")
    axes[0].scatter(x_train, y_train, color="black", s=18, label="training points")
    axes[0].set(title="Polynomial models", xlabel="x", ylabel="y")
    axes[0].legend(fontsize=8)
    axes[1].plot(degrees, train_errors, "o-", label="training error")
    axes[1].plot(degrees, test_errors, "o-", label="unseen test error")
    axes[1].set(title="Fit versus generalization", xlabel="polynomial degree", ylabel="mean squared error")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print("Supervised learning means examples include the correct answer (label). Here, y is the answer.")
    print("A degree-1 curve may underfit; a degree-9 curve may overfit noise; compare test errors to choose.")
    for d, tr, te in zip(degrees, train_errors, test_errors):
        print(f"degree {d}: train error={tr:.3f}, test error={te:.3f}")
    print(f"Saved the Unit 1 teaching plot: {OUT}")


def discrete_probability(data):
    heading("3. Discrete random variables, probability rules, and Bayes")
    d = data[["Target_Direction", "RSI", "Volatility_20D"]].dropna().copy()
    d["Direction"] = np.where(d["Target_Direction"] == 1, "Up", "Down")
    d["RSI_Regime"] = pd.cut(d["RSI"], [-np.inf, 30, 70, np.inf],
                             labels=["Oversold", "Neutral", "Overbought"])
    joint = pd.crosstab(d["Direction"], d["RSI_Regime"], normalize="all")
    print("A discrete random variable has a small set of possible categories.")
    print("Joint probabilities P(direction, RSI regime):")
    print(joint.round(3).to_string())
    direction_probability = d["Direction"].value_counts(normalize=True).sort_index()
    print("\nSum rule: add across RSI categories to get P(direction):")
    print(direction_probability.round(3).to_string())

    # Product rule check: P(direction, RSI) = P(direction | RSI) P(RSI).
    conditional = pd.crosstab(d["Direction"], d["RSI_Regime"], normalize="columns")
    rsi_probability = d["RSI_Regime"].value_counts(normalize=True).reindex(joint.columns)
    reconstructed = conditional.mul(rsi_probability, axis=1).reindex_like(joint)
    print(f"Product rule check: largest reconstruction difference = {(joint-reconstructed).abs().to_numpy().max():.6f}")

    bayes = conditional.loc["Up"] * rsi_probability / direction_probability["Up"]
    print("Bayes' rule: P(RSI regime | next day is up):")
    print("P(RSI | Up) = P(Up | RSI) * P(RSI) / P(Up)")
    print(bayes.round(3).to_string())
    print("This is the probability of an RSI category after we know the direction label.")

    pooled_table = pd.crosstab(d["Direction"], d["RSI_Regime"])
    chi2, p, _, _ = chi2_contingency(pooled_table)
    print(f"\nIndependence test, all rows: chi-square={chi2:.2f}, p={p:.4f}.")
    d["Volatility_Regime"] = np.where(d["Volatility_20D"] > d["Volatility_20D"].median(), "High", "Low")
    for regime, subset in d.groupby("Volatility_Regime", observed=True):
        table = pd.crosstab(subset["Direction"], subset["RSI_Regime"])
        p_value = chi2_contingency(table)[1]
        conclusion = "evidence of association" if p_value < 0.05 else "not enough evidence of association"
        print(f"Conditional on {regime.lower()} volatility: p={p_value:.4f} ({conclusion}).")
    print("A p-value above 0.05 does not prove independence; it means this test did not find strong evidence against it.")


def continuous_probability(data):
    heading("4. Continuous random variables, density, quantiles, mean and variance")
    ticker = data[data["Ticker"] == "AAPL"]
    returns = ticker["Return_1D"].dropna().to_numpy()
    mean, variance = returns.mean(), returns.var(ddof=1)
    q05, q95 = np.quantile(returns, [0.05, 0.95])
    degrees, loc, scale = stats.t.fit(returns)
    print("A continuous random variable can take many numeric values; here X is AAPL's daily return.")
    print(f"Mean / sample expectation E[X] = {mean:.6f} ({mean*100:.3f}% daily average)")
    print(f"Variance = {variance:.8f}; standard deviation = {np.sqrt(variance):.4f} ({np.sqrt(variance)*100:.2f}%)")
    print(f"Empirical 5th and 95th percentiles = {q05*100:.2f}% and {q95*100:.2f}%")
    print("A probability density describes likelihood per unit of return; area under it over a range is probability.")
    print(f"Student-t fit: degrees of freedom={degrees:.2f}; its heavy tails allow more extreme returns than a Normal curve.")
    print("Expectation is the probability-weighted average; variance measures spread around that average.")
    xx = np.linspace(np.quantile(returns, 0.005), np.quantile(returns, 0.995), 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(returns, bins=55, density=True, alpha=0.45, label="observed returns")
    ax.plot(xx, stats.t.pdf(xx, degrees, loc, scale), label="fitted Student-t density")
    ax.set(title="AAPL daily returns", xlabel="daily return", ylabel="density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT.replace("unit1_summary", "unit1_returns"), dpi=140)
    plt.close(fig)


def covariance_lesson(data):
    heading("5. Expectation and covariance across stocks")
    returns = data.pivot(index="Date", columns="Ticker", values="Return_1D").dropna()
    covariance = returns.cov()
    correlation = returns.corr()
    weights = np.repeat(1 / len(returns.columns), len(returns.columns))
    portfolio = returns.to_numpy() @ weights
    print("Covariance measures how two stocks' returns vary together; correlation is its standardized version (-1 to +1).")
    print("Sample covariance matrix:")
    print(covariance.round(6).to_string())
    print("Correlation matrix:")
    print(correlation.round(2).to_string())
    print(f"Expected equal-weight daily portfolio return = {portfolio.mean():.5f}")
    print(f"Portfolio daily volatility = {portfolio.std(ddof=1)*100:.2f}%")
    print("Because stocks are not perfectly correlated, combining them can reduce portfolio volatility.")


def run(data):
    print("\nUNIT 1 - ML INTRODUCTION AND PROBABILITY")
    machine_learning_intro(data)
    polynomial_lesson()
    discrete_probability(data)
    continuous_probability(data)
    covariance_lesson(data)
    print("\nUnit 1 complete. Look at results/unit1_summary.png and results/unit1_returns.png")
