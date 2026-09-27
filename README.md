# Stock Market Regime Analysis and Short-Term Movement Prediction

A stock-market ML project that's built up *syllabus unit by syllabus unit* —
every concept you're taught gets implemented against real market data
(AAPL, MSFT, GOOGL, AMZN, NVDA), not toy textbook numbers, so you can see
what each idea actually looks like when the data is messy and real.

## Status

- [x] Milestone 1 — Data pipeline (download, clean, features, targets, split)
- [x] Milestone 2 — Unit 1: intro to ML, polynomial curve fitting, probability theory
- [x] Milestone 3 & 4 — **Unit 2: linear models for regression & classification** ← you are here

Both syllabus units (Unit 1 and Unit 2, in full) are now implemented. See
below for how to run everything and what to expect.

---

## 1. Setup

```bash
python -m venv venv
```
Activate it — **this step trips people up more than anything else in this
repo, see the gotcha below**:
```bash
source venv/bin/activate        # Mac/Linux
venv\Scripts\Activate.ps1        # Windows PowerShell
venv\Scripts\activate.bat        # Windows cmd.exe
```
Then install dependencies:
```bash
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
Four short "labs" mapped to Unit 1 of the syllabus, printing worked numeric
examples and saving plots under `results/unit1/`. Takes under a minute.

## 4. Run Unit 2 (Milestones 3 & 4)

```bash
python run_unit2.py
```
Eight short "labs" — four on regression, four on classification — mapped to
Unit 2 of the syllabus, printing worked numeric examples and saving plots
under `results/unit2/`. Takes 1-2 minutes (the SVM/kernel step is the slow
part).

---

## What Unit 1 covers, and where to find it

| Syllabus topic | File | What it does |
|---|---|---|
| What is ML, why; supervised learning; polynomial curve fitting | [`src/unit1/polynomial_curve_fitting.py`](src/unit1/polynomial_curve_fitting.py) | Fits polynomials of increasing degree to (a) the classic textbook `sin(2πx) + noise` example and (b) a real 30-day AAPL price window. Shows training error falling monotonically while held-out error forms a U — the precise, non-hand-wavy definition of **underfitting vs. overfitting**. |
| Discrete random variables; fundamental rules (sum/product); Bayes' rule; independence & conditional independence | [`src/unit1/discrete_probability.py`](src/unit1/discrete_probability.py) | Treats `Target_Direction` (up/down) as a Bernoulli RV and an RSI-based regime (Oversold/Neutral/Overbought) as a second discrete RV. Builds the joint distribution, verifies the **sum rule** and **product rule** numerically, applies **Bayes' rule** by hand, then runs a chi-square **independence test** — first pooled, then *stratified by volatility regime* to show **conditional independence** can differ from unconditional independence. |
| Continuous random variables; probability densities; quantiles; mean & variance; expectation | [`src/unit1/continuous_probability.py`](src/unit1/continuous_probability.py) | Treats AAPL's daily return as a continuous RV. Computes mean/variance/skew/kurtosis, fits a **Normal** and a **Student-t density** via maximum likelihood (t wins — real returns have fat tails), reads off **quantiles** (literally what "Value at Risk" is), evaluates a density as an *area*, and computes **expectation** two ways to confirm they agree. |
| Expectation and covariance (multivariate) | [`src/unit1/covariance_expectation.py`](src/unit1/covariance_expectation.py) | Builds the 5×5 **covariance/correlation matrix** across tickers, checks it's symmetric and positive semi-definite, then computes portfolio expectation/variance two ways to show diversification falls straight out of the covariance-matrix algebra. |

`run_unit1.py` calls all four in order.

## What Unit 2 covers, and where to find it

Every Unit 2 module shares one loader (`src/unit2/data_utils.py`): it pools
all 5 tickers, turns absolute-price moving averages into scale-free ratios
(`Price_to_MA5` etc. — otherwise AAPL's price scale would dominate a pooled
model), splits chronologically via `src/data/split.py`, and **fits any
scaler strictly on the training split only** — the exact leakage pitfall
called out in the original project review. Regression predicts
`Target_Return`; classification predicts `Target_Direction`.

### Regression (predicting next-day return)

| Syllabus topic | File | What it does |
|---|---|---|
| Maximum likelihood estimation, least squares | [`least_squares.py`](src/unit2/regression/least_squares.py) | Solves the normal equations by hand, proves the result is numerically identical to sklearn's `LinearRegression` — the concrete demonstration that **MLE under Gaussian noise = least squares**, not just "similar to." |
| Robust linear regression | [`robust_regression.py`](src/unit2/regression/robust_regression.py) | Compares OLS vs. Huber-loss regression, then deliberately injects synthetic outliers into the training targets and tracks how far each model's coefficients drift — OLS drifts ~5x more than Huber. |
| Ridge regression | [`ridge_regression.py`](src/unit2/regression/ridge_regression.py) | Closed-form ridge solution verified against sklearn, a full shrinkage path (every coefficient sliding to 0 as λ grows), and λ selected properly using the **validation** split, never touching test. |
| Bayesian linear regression | [`bayesian_linear_regression.py`](src/unit2/regression/bayesian_linear_regression.py) | Closed-form Gaussian posterior over the weights. A toy demo shows predictive uncertainty visibly shrinking as N grows (ties back to Unit 1's polynomial curve fitting); on real data, proves the **posterior mean equals the ridge solution** for λ=α/β, and checks predictive-interval calibration. |

### Classification (predicting next-day up/down)

| Syllabus topic | File | What it does |
|---|---|---|
| Discriminant function; probabilistic generative models | [`generative_models.py`](src/unit2/classification/generative_models.py) | Fits class-conditional Gaussians with **shared covariance**, derives the linear discriminant `w, w0` by hand, and confirms it's numerically identical (cosine similarity = 1.0) to sklearn's `LinearDiscriminantAnalysis`. Plots the resulting linear decision boundary. |
| Probabilistic discriminative models | [`discriminative_models.py`](src/unit2/classification/discriminative_models.py) | Implements logistic regression MLE fitting via hand-written **IRLS** (Newton-Raphson), verified against sklearn's `LogisticRegression`. Directly compares this discriminative model against the previous generative one on the identical test set (accuracy, precision, recall, ROC-AUC). |
| Laplace approximation; Bayesian logistic regression | [`laplace_bayesian_logistic.py`](src/unit2/classification/laplace_bayesian_logistic.py) | Approximates the posterior over logistic-regression weights as a Gaussian centered at the MAP estimate (Hessian = posterior precision), then applies Bishop's probit correction so **predictive probabilities are pulled toward 0.5 in proportion to how uncertain the model is** — a toy 2D demo makes this dramatically visible. |
| Kernel functions; using kernels in GLMs; kernel trick; SVMs | [`kernels_svm.py`](src/unit2/classification/kernels_svm.py) | Builds an explicit polynomial feature map and proves it gives numerically identical inner products to the polynomial *kernel function* (the kernel trick, made concrete) — then uses a kernel inside a GLM (kernel ridge regression) and compares linear vs. RBF-kernel SVMs, including a 2D decision-boundary plot. |

`run_unit2.py` calls all eight in order.

---

## What to look for in the output (so you know it "worked")

**Unit 1:**
- **Polynomial demo:** `M=9` on 10 synthetic points should look like a wild, wiggly curve hitting every dot exactly (train RMSE ≈ 0) — that's overfitting. Train-vs-test error should form a U with a clear best degree in between. On real AAPL data, a *low* best degree (1-3) is the expected, healthy result.
- **Discrete probability:** every "check" line should print a difference of `0.000000` — these are exact identities.
- **Continuous probability:** Student-t should beat Normal on log-likelihood; excess kurtosis should be clearly above 0.
- **Covariance:** smallest eigenvalue must be ≥ 0 (or ~`-1e-18`, floating-point noise).

**Unit 2:**
- **Least squares:** the MLE/normal-equations/sklearn triple-check should print a difference of essentially `0`. Test R² near 0 (or slightly negative) is the *correct*, honest outcome for next-day returns.
- **Robust regression:** OLS coefficient drift should grow much faster than Huber's as synthetic outlier contamination increases.
- **Ridge:** the shrinkage-path plot should show every line sliding toward 0 as λ (x-axis) grows. Don't be surprised if the validation search picks the *largest* λ tried — see the gotcha below on what that means.
- **Bayesian linear regression:** the toy plot's uncertainty band should visibly narrow from N=2 to N=25. On real data, the posterior-mean-vs-ridge check should print a difference of essentially `0`, and roughly ~95% of test targets should fall inside the 95% predictive interval (a little under 95% is expected — real returns are fatter-tailed than the Gaussian likelihood assumes).
- **Generative model:** cosine similarity vs. sklearn's LDA should print `1.000000` (or `-1.000000`, same boundary).
- **Discriminative model:** hand-written IRLS should land close to sklearn's unregularized logistic regression. Accuracy in the low-to-mid 50s on test is expected and correct, not a bug.
- **Laplace/Bayesian logistic:** the toy 2D contour plot is the one to actually look at — the "Bayesian" panel should look visibly softer/less saturated than the "plug-in" panel, especially away from the training clusters. On the real data (thousands of rows, only 12 features), the correction is nearly invisible — that's expected; see the gotcha below.
- **Kernels/SVM:** the polynomial-feature-map-vs-kernel-function check should print a difference of essentially `0`. The RBF-kernel SVM boundary plot may look like it's carving out small "islands" — see the gotcha below on why that's a lesson, not a bug.

---

## Known gotchas & things to watch out for

- **"ModuleNotFoundError: No module named 'X'" means the venv isn't active.** By far the most common error you'll hit: running `python main.py` (or any script here) with your system Python instead of the project's `venv` will fail with a missing-module error even though `pip install -r requirements.txt` succeeded earlier — it succeeded *inside the venv*, which isn't the interpreter that ran. Fix: activate the venv every new terminal session (`venv\Scripts\Activate.ps1` on Windows PowerShell) before running anything, or just call the venv's Python directly: `.\venv\Scripts\python.exe main.py`. If PowerShell refuses to run the activation script citing execution policy, use `powershell -ExecutionPolicy Bypass -File .\venv\Scripts\Activate.ps1`, or the direct-call form above (no activation needed at all).
- **Windows console + special characters.** `print()`-ing `π`, `·`, em-dashes, or `⟂` crashes with `UnicodeEncodeError` on a default Windows terminal (cp1252 codepage). Every string printed anywhere in this repo is kept plain-ASCII on purpose. If you add your own prints with fancy Unicode, either keep them ASCII or run `chcp 65001` first.
- **The Adjusted-Close bug (already fixed, but know why).** An earlier version computed returns off raw `Close` instead of `Adj Close`. Four of five tickers split during 2015-2025 (AAPL 4:1 2020, NVDA 4:1 2021 + 10:1 2024, GOOGL/AMZN 20:1 2022) — unadjusted close shows a split as a fake single-day price crash, which would land directly in the target column. Fixed in `src/features/engineer.py`. **Any new price-based feature must use `Adj Close`.**
- **Survivorship bias.** The five tickers are exactly the 2015-2025 mega-cap tech "winners." Any "the market tends to do X" claim from this data is really a claim about five specific successful companies. Say this out loud in any write-up.
- **Scale-free features matter once you pool tickers (Unit 2).** `Price_to_MA5/20/50` and `Volume_to_MA` exist in `data_utils.py` specifically because AAPL's raw price and GOOGL's raw price live on totally different scales — a pooled model trained on raw `MA_50` would mostly learn "which ticker is this" instead of a real momentum signal.
- **Ridge's validation search may pick the largest λ in the grid.** If `select_alpha_on_validation` prints a note that it hit the edge of `ALPHA_GRID`, that means the validation set prefers shrinking weights all the way toward "just predict the mean" — a real, honest signal that there's essentially no exploitable linear relationship, not a bug in the search.
- **The Laplace correction can look negligible on real data.** With ~8,500 training rows and only 12 features, the posterior over the weights is already tight, so the plug-in and Bayesian probabilities barely differ. That's expected — the effect is far more visible in the toy 2D demo (only 40 points) in the same module, and would matter far more on any real project with fewer rows per parameter.
- **The RBF-kernel SVM plot might show small "islands" of the minority class.** That's the flexible kernel latching onto a handful of noisy points rather than finding real structure — it's the SVM analogue of the Unit 1 degree-9 polynomial overfitting demo, not a sign the kernel was misconfigured.
- **Kernel methods are subsampled to ~1,500 training rows** (`KERNEL_TRAIN_SAMPLE` in `kernels_svm.py`) purely for runtime — kernel matrices scale roughly O(n²)-O(n³). Test/val evaluation always uses the full split, so reported metrics are still honest; only the *fit* is on a subsample.
- **yfinance can be rate-limited or flaky.** Just rerun `python main.py` — `download.py` retries automatically.
- **RSI can divide by zero** on flat/all-up stretches; handled (produces `NaN`, not a crash) — expected behavior.
- **~50 rows at the start of each ticker's history get dropped** (need 50 prior days for `MA_50`). Expected, not data loss.
- **Chi-square p-values are not proof of independence** — a p-value above 0.05 means "no evidence found against independence," a weaker claim than "these are independent."
- **Set realistic accuracy expectations everywhere in Unit 2.** Predicting next-day stock direction/return from OHLCV-derived features is close to the textbook definition of a hard problem (efficient market hypothesis). Accuracy in the low-to-mid 50s / R² near 0 is a *correct* result throughout this project, not a failure — resist tuning until you hit 90%, because that almost always means leakage crept back in.
- **Run things from the repo root.** All scripts use relative paths like `data/processed/...`, so `python run_unit2.py` needs to run from this folder, not from `src/`.

---

## Project layout

```
main.py                     # Milestone 1: download -> clean -> engineer features
run_unit1.py                  # Milestone 2: runs all four Unit 1 modules
run_unit2.py                  # Milestones 3-4: runs all eight Unit 2 modules
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
  unit2/
    data_utils.py                # pooled loader, scale-free features, leakage-safe scaling/split
    regression/
      least_squares.py             # MLE == OLS via normal equations
      robust_regression.py         # Huber vs OLS under outlier contamination
      ridge_regression.py          # closed-form ridge, shrinkage path, validation-selected lambda
      bayesian_linear_regression.py  # closed-form posterior, predictive uncertainty
    classification/
      generative_models.py         # shared-covariance Gaussian discriminant (LDA)
      discriminative_models.py     # logistic regression via hand-written IRLS
      laplace_bayesian_logistic.py # Laplace approximation, Bayesian predictive probabilities
      kernels_svm.py                # kernel trick, kernel ridge regression, linear/RBF SVM
data/
  raw/                          # one CSV per ticker, untouched once downloaded
  processed/                    # cleaned + feature-engineered CSVs
results/
  unit1/
    polynomial/                 # curve-fitting figures
    probability/                # density/covariance figures + a text summary
  unit2/
    regression/                  # OLS/ridge/robust/Bayesian regression figures
    classification/              # generative/discriminative/Laplace/SVM figures
```

## What's next

Both syllabus units are now fully implemented end-to-end: data pipeline →
probability foundations → linear regression models → linear classification
models, all evaluated honestly on chronologically held-out real market
data. From here, natural next steps (not currently planned as a milestone,
just food for thought) would be walk-forward/expanding-window validation
instead of a single fixed split, and extending the feature set or ticker
universe to address the survivorship-bias limitation noted above.
