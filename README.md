# Stock Market Regime Analysis and Short-Term Movement Prediction

A stock-market ML project that's built up *syllabus unit by syllabus unit* —
every concept you're taught gets implemented against real market data
(AAPL, MSFT, GOOGL, AMZN, NVDA), not toy textbook numbers, so you can see
what each idea actually looks like when the data is messy and real.

## Status

- [x] Milestone 1 — Data pipeline (download, clean, features, targets, split)
- [x] Milestone 2 — **Unit 1: intro to ML, polynomial curve fitting, probability theory** ← you are here
- [ ] Milestone 3 — Unit 2: linear models for regression (MLE, ridge, Bayesian linear regression)
- [ ] Milestone 4 — Unit 2: linear models for classification (discriminant functions, logistic regression, kernels, SVMs)

---

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the data pipeline (Milestone 1)

```bash
python main.py
```

This will:
1. Download daily OHLCV data for AAPL, MSFT, GOOGL, AMZN, NVDA (2015-01-01 to 2025-01-01) into `data/raw/`.
2. Clean it (dedupe, sort, fix invalid rows) into `data/processed/<TICKER>_clean.csv`.
3. Engineer features + targets into `data/processed/<TICKER>_features.csv`.

You can also run each step individually:

```bash
python src/data/download.py
python src/data/clean.py
python src/features/engineer.py
```

**Sanity checks before moving on:** open a processed feature file and confirm:
- [ ] No missing values remain (`df.isna().sum()` should be all zeros)
- [ ] `Date` is sorted and has no duplicates
- [ ] Row count roughly matches expected trading days (~252/year × years, minus the 50-day warm-up)
- [ ] `Target_Return` for row *t* really is tomorrow's return, not today's (spot-check a few rows by hand)
- [ ] Use `src/data/split.py`'s `time_series_split()` — never `sklearn.train_test_split()` on this data

## 3. Run Unit 1 (Milestone 2)

```bash
python run_unit1.py
```

This runs four short "labs," each one mapped to a chunk of your Unit 1
syllabus, printing worked numeric examples to the console and saving
plots under `results/unit1/`. It takes under a minute.

---

## What Unit 1 actually covers, and where to find it

| Syllabus topic | File | What it does |
|---|---|---|
| What is ML, why; supervised learning; polynomial curve fitting | [`src/unit1/polynomial_curve_fitting.py`](src/unit1/polynomial_curve_fitting.py) | Fits polynomials of increasing degree to (a) the classic textbook `sin(2πx) + noise` example and (b) a real 30-day AAPL price window. Shows training error falling monotonically while held-out error forms a U — the precise, non-hand-wavy definition of **underfitting vs. overfitting**. |
| Discrete random variables; fundamental rules (sum/product); Bayes' rule; independence & conditional independence | [`src/unit1/discrete_probability.py`](src/unit1/discrete_probability.py) | Treats `Target_Direction` (up/down) as a Bernoulli RV and an RSI-based regime (Oversold/Neutral/Overbought) as a second discrete RV. Builds the joint distribution, verifies the **sum rule** and **product rule** numerically, applies **Bayes' rule** by hand and checks it against direct counting, then runs a chi-square **independence test** — first pooled, then *stratified by volatility regime* to show how **conditional independence** can differ from unconditional independence. |
| Continuous random variables; probability densities; quantiles; mean & variance; expectation | [`src/unit1/continuous_probability.py`](src/unit1/continuous_probability.py) | Treats AAPL's daily return as a continuous RV. Computes mean/variance/skew/kurtosis, fits a **Normal** and a **Student-t density** via maximum likelihood (and shows the t-distribution wins — real returns have fat tails), reads off **quantiles** (which is literally what "Value at Risk" is), evaluates a density as an *area* (`P(a ≤ X ≤ b)` via CDF difference, not a point probability), and computes **expectation** two independent ways (sample mean vs. numerically integrating `x·f(x)`) to show they agree. |
| Expectation and covariance (multivariate) | [`src/unit1/covariance_expectation.py`](src/unit1/covariance_expectation.py) | Builds the 5×5 **covariance/correlation matrix** across all tickers' daily returns, checks it's symmetric and positive semi-definite, then computes an equal-weight portfolio's expected return and variance two ways — the closed-form identities `E[P] = w·E[R]` and `Var[P] = wᵀ Cov(R) w`, versus just measuring the realized portfolio series directly — to show diversification isn't a vibe, it falls straight out of the covariance-matrix algebra. |

`run_unit1.py` just calls all four of the above in order, with section headers, and tells you where the output plots landed.

---

## What to look for in the output (so you know it "worked")

- **Polynomial demo:** the `M=9` fit on 10 synthetic points should look like a wild, wiggly curve that hits every dot exactly (train RMSE ≈ 0) but flies off wildly between points — that's the overfitting picture. The train-vs-test error plot should show test error going down then back up, with a clear best degree in between (usually 3–5 on the synthetic data). On the real AAPL window, expect the *best* degree to be low (1–3) — that's a **good** result, not a disappointing one; it means the held-out check is doing its job.
- **Discrete probability:** every "check" line should print a difference of `0.000000` — sum rule, product rule, and Bayes' rule are exact identities, so if they don't match to floating-point precision, something is broken (not "close enough," genuinely broken).
- **Continuous probability:** Student-t should have a higher log-likelihood than Normal (fatter tails fit financial returns better — this is one of the most consistent stylized facts in all of finance). Excess kurtosis should be well above 0.
- **Covariance:** the printed "smallest eigenvalue" must be ≥ 0 (or a tiny negative number like `-1e-18`, which is just floating-point noise) — a real covariance matrix can never have a negative eigenvalue. Correlations between the 5 tickers should all be positive (they're all large-cap tech, so they tend to move together) but well below 1.0.

---

## Known gotchas & things to watch out for

- **Windows console + special characters.** `print()`-ing `π`, `·`, em-dashes, or `⟂` will crash with a `UnicodeEncodeError` on a default Windows terminal (cp1252 codepage), even though the same characters render fine in a Jupyter notebook or on Mac/Linux. Every string printed to the console in this repo is kept plain-ASCII on purpose (`pi` instead of `π`, `-` instead of `—`, etc.) for exactly this reason. If you add your own `print()` statements with fancy Unicode, either keep them ASCII or run `chcp 65001` first to switch your terminal to UTF-8.
- **The Adjusted-Close bug (already fixed, but know why).** An earlier version of `engineer.py` computed returns off the raw `Close` column instead of `Adj Close`. Four of these five tickers had a stock split during 2015–2025 (AAPL 4:1 in 2020, NVDA 4:1 in 2021 + 10:1 in 2024, GOOGL & AMZN both 20:1 in 2022), and a raw, unadjusted close price shows a split as an enormous fake single-day price crash. That fake crash would land directly in `Target_Return`/`Target_Direction` and quietly poison the label column. This is now fixed — `add_features()` in `src/features/engineer.py` explicitly uses `Adj Close`. **If you ever add a new feature that touches price, make sure it uses `Adj Close`, not `Close`.**
- **Survivorship bias.** The five tickers (AAPL/MSFT/GOOGL/AMZN/NVDA) are exactly the 2015–2025 mega-cap tech "winners." Any claim like "the market tends to do X" based on this data is really a claim about five specific large, successful companies, not "the market." This isn't something to fix — it's something to say out loud in any write-up or presentation.
- **yfinance can be rate-limited or flaky.** If a ticker download fails, just rerun `python main.py` — `download.py` retries automatically, but Yahoo occasionally needs a minute to stop complaining.
- **RSI can divide by zero** when there's no losing day in the lookback window. This is handled (0 replaced with NaN before dividing), so you'll see occasional `NaN` RSI on flat/all-up stretches — that's correct behavior, not a bug.
- **~50 rows at the start of each ticker's history get dropped** because `MA_50` needs 50 prior days to exist before it's defined. Expected, not data loss.
- **Chi-square p-values are not proof of independence.** A p-value above 0.05 in `discrete_probability.py` means "we didn't find strong evidence *against* independence," which is a much weaker (and more honest) claim than "these are independent." Don't over-claim this in a report.
- **Sample sizes get small once you stratify.** The conditional-independence check splits the data by volatility regime, which roughly halves the sample each time. If you stratify by a third or fourth variable on top of that, your chi-square test can lose statistical power fast — watch the printed `n=` before trusting a "fail to reject" result.
- **Set realistic accuracy expectations for later units.** Predicting next-day stock direction from OHLCV-derived features is close to the textbook definition of a hard problem (see: efficient market hypothesis). When Unit 2's classifiers land, an honest 52–56% test accuracy is a *correct* result, not a failure — resist the urge to keep tuning until you hit 90%, because that almost always means leakage snuck back in somewhere.
- **Run things from the repo root.** All scripts use relative paths like `data/processed/...`, so `python run_unit1.py` (or `python main.py`) needs to be run from this folder, not from `src/` or `notebooks/`.

---

## Project layout

```
main.py                     # Milestone 1: download -> clean -> engineer features
run_unit1.py                 # Milestone 2: runs all four Unit 1 modules
src/
  data/
    download.py               # pulls raw OHLCV from yfinance
    clean.py                  # dedupe, sort, drop invalid rows
    split.py                  # chronological train/val/test split (never shuffle!)
  features/
    engineer.py                # returns, moving averages, volatility, RSI, MACD, targets
  unit1/
    polynomial_curve_fitting.py  # curve fitting, over/underfitting
    discrete_probability.py      # discrete RVs, sum/product rule, Bayes, independence
    continuous_probability.py    # continuous RVs, densities, quantiles, expectation
    covariance_expectation.py    # multivariate expectation & covariance, portfolio variance
  classification/ probability/ regression/   # empty stubs, reserved for Unit 2 (Milestones 3-4)
data/
  raw/                          # one CSV per ticker, untouched once downloaded
  processed/                    # cleaned + feature-engineered CSVs
results/
  unit1/
    polynomial/                 # curve-fitting figures
    probability/                # density/covariance figures + a text summary
```

## What's next (Unit 2)

Once you're comfortable with everything above, Milestone 3 will build linear
regression models on top of this same feature set: maximum likelihood /
least squares, robust regression, ridge regression, and Bayesian linear
regression — all predicting `Target_Return`. Milestone 4 will then build
classifiers predicting `Target_Direction`: generative and discriminative
probabilistic models, Laplace approximation, Bayesian logistic regression,
kernel methods, and SVMs.
