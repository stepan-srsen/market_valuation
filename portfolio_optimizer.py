"""Portfolio optimizer: maximize the Sharpe ratio across MSCI World factor indices, gold, silver and Nasdaq 100.

Loads historical monthly prices for:
  - MSCI World, MSCI World Value, MSCI World Momentum, MSCI World Energy, MSCI World High Dividend Yield (xlsx)
  - Gold futures, Silver futures, Nasdaq 100 index (yfinance, cached locally)

Aligns all series to their maximum common date range, finds the long-only weights that
maximize the Sharpe ratio (mean-variance optimization), and reports performance,
volatility, Sharpe ratio and maximum drawdown for the optimal portfolio and its constituents.
"""
import datetime as dt
import glob
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# yfinance tickers loaded alongside the MSCI factor indices, as (ticker, series name) pairs.
YFINANCE_TICKERS = [
    ("GC=F", "Gold"),
    # ("SI=F", "Silver"), # from long-term perspective, silver is highly correlated with gold but more volatile
    ("^NDX", "Nasdaq 100"),
    # ("^GSPC", "S&P 500"), # MSCI world is similar but more diversified than S&P 500
]

# Partial names matched (as a substring) against each cached MSCI xlsx filename to select which
# factor indices to load. Comment out entries to exclude them from the optimization.
# the files from msci.com have to be present in the data_cache folder as they cannot be downloaded automatically and no proxy has long enough history
MSCI_INDICES = [
#    "MSCI World Index", # from long-term perspective, highly correlated with MSCI World Momentum and Dividend but with slightly worse Sharpe ratio
#    "MSCI World Value Index", # from long-term perspective, highly correlated with MSCI World High Dividend but with slightly worse Sharpe ratio
    "MSCI World Momentum Index",
    "MSCI World Energy Index",
    "MSCI World High Dividend Yield Index",
]

# Annualized risk-free rate assumption used in the Sharpe ratio calculation.
RISK_FREE_RATE = 0.0
MONTHS_PER_YEAR = 12

# Quantiles of the excess CAPE yield used to split history into valuation regimes.
DEFAULT_QUANTILES = [1/2]
#DEFAULT_QUANTILES = (1/3, 2/3)
# DEFAULT_QUANTILES = (1/4, 2/4, 3/4)
#DEFAULT_QUANTILES = (1/5, 2/5, 3/5, 4/5)


# Block bootstrap settings used to estimate confidence intervals around the optimal portfolios.
BOOTSTRAP_BLOCK_SIZE = 12
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI_LEVEL = 0.90
# Fewer iterations for the quantile-count scan, which reruns the bootstrap once per quantile count.
BOOTSTRAP_SCAN_ITERATIONS = 200

def load_msci_index(path: Path) -> pd.Series:
    """Load a monthly MSCI index export (xlsx) and return a Series indexed by month Period."""
    df = pd.read_excel(path, header=5, usecols=[0, 1])
    df.columns = ["Date", "Price"]
    df = df.dropna(subset=["Date", "Price"])
    df["Date"] = pd.to_datetime(df["Date"])
    name = Path(path).name.split(" - ")[1]  # e.g. "MSCI World Value Index"
    series = pd.Series(df["Price"].values, index=df["Date"].dt.to_period("M"), name=name)
    return series


def fetch_yfinance_monthly(ticker: str, name: str) -> pd.Series:
    """Download daily closes for ticker via yfinance (cached locally, refreshed once per day) and return monthly closes."""
    cache_path = CACHE_DIR / f"{ticker.replace('^', '').replace('=', '_')}.pkl"

    should_download = True
    if cache_path.exists():
        file_mod_time = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_mod_time.date() == dt.datetime.now().date():
            should_download = False

    if should_download:
        data = yf.download(ticker, auto_adjust=True, period="max", interval="1d")
        if data.empty:
            raise ValueError(f"No data returned for {ticker}.")
        data.to_pickle(cache_path)
    else:
        data = pd.read_pickle(cache_path)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"][ticker].dropna()
    else:
        closes = data["Close"].dropna()

    monthly = closes.groupby(closes.index.to_period("M")).last()
    monthly.name = name
    return monthly


def load_excess_cape_yield(path: Path = DATA_DIR / "ie_data.xls") -> pd.Series:
    """Load Shiller's monthly excess CAPE yield (in %) and return a Series indexed by month Period."""
    df = pd.read_excel(path, sheet_name="Data", header=7)
    df = df.dropna(subset=["Date", "Yield"])
    year = df["Date"].astype(int)
    month = ((df["Date"] - year) * 100).round().astype(int)  # dates are encoded as YYYY.MM
    period = pd.PeriodIndex.from_fields(year=year, month=month, freq="M")
    return pd.Series(df["Yield"].values * 100, index=period, name="Excess CAPE Yield")


def load_all_prices() -> pd.DataFrame:
    """Load all constituent price series and align them to their maximum common date range."""
    msci_paths = sorted(glob.glob(str(CACHE_DIR / "*MSCI*.xlsx")))
    msci_paths = [p for p in msci_paths if any(name in Path(p).name for name in MSCI_INDICES)]
    series = [load_msci_index(p) for p in msci_paths]
    series.extend(fetch_yfinance_monthly(ticker, name) for ticker, name in YFINANCE_TICKERS)

    prices = pd.concat(series, axis=1).sort_index()
    prices = prices.dropna(how="any")  # keep only the maximum common date range
    prices.index = prices.index.to_timestamp(how="end").normalize()
    return prices


def portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> tuple[float, float]:
    """Return annualized (return, volatility) for a given weight vector."""
    port_return = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(weights @ cov_matrix.values @ weights))
    return port_return, port_vol


def negative_sharpe(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
    port_return, port_vol = portfolio_stats(weights, mean_returns, cov_matrix)
    return -(port_return - RISK_FREE_RATE) / port_vol


def optimize_sharpe(returns: pd.DataFrame) -> pd.Series:
    """Find the long-only weights that maximize the Sharpe ratio."""
    mean_returns = returns.mean() * MONTHS_PER_YEAR
    cov_matrix = returns.cov() * MONTHS_PER_YEAR

    n = len(mean_returns)
    init_guess = np.repeat(1.0 / n, n)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    result = minimize(
        negative_sharpe,
        init_guess,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return pd.Series(result.x, index=returns.columns, name="Weight")


def optimize_linear_weights(returns: pd.DataFrame, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Fit each asset weight as w_i(t) = a_i + b_i * x(t), long-only and summing to 1 in every
    month of the sample, maximizing the annualized Sharpe ratio of the resulting portfolio.

    Returns (intercepts, slopes), each a Series indexed by asset.
    """
    assets = returns.columns
    n = len(assets)
    x_values = x.to_numpy()
    r_values = returns.to_numpy()

    def weights_matrix(params: np.ndarray) -> np.ndarray:
        a, b = params[:n], params[n:]
        return a[None, :] + b[None, :] * x_values[:, None]  # shape (T, n)

    def neg_sharpe(params: np.ndarray) -> float:
        port_returns = (weights_matrix(params) * r_values).sum(axis=1)
        mean_ann = port_returns.mean() * MONTHS_PER_YEAR
        vol_ann = port_returns.std() * np.sqrt(MONTHS_PER_YEAR)
        return -(mean_ann - RISK_FREE_RATE) / vol_ann

    init_guess = np.concatenate([np.repeat(1.0 / n, n), np.zeros(n)])
    constraints = (
        {"type": "eq", "fun": lambda p: np.sum(p[:n]) - 1.0},
        {"type": "eq", "fun": lambda p: np.sum(p[n:])},
        {"type": "ineq", "fun": lambda p: weights_matrix(p).reshape(-1)},  # w(t) >= 0 every month
        {"type": "ineq", "fun": lambda p: 1.0 - weights_matrix(p).reshape(-1)},  # w(t) <= 1 every month
    )

    result = minimize(neg_sharpe, init_guess, method="SLSQP", constraints=constraints, options={"maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"Linear weight optimization failed: {result.message}")
    a = pd.Series(result.x[:n], index=assets, name="Intercept")
    b = pd.Series(result.x[n:], index=assets, name="Slope")
    return a, b


def linear_weights_matrix(a: pd.Series, b: pd.Series, x: pd.Series) -> pd.DataFrame:
    """Weights w_i(t) = a_i + b_i * x(t), clipped to [0, 1] and renormalized to sum to 1 per month.

    Clipping is a no-op for the in-sample x the coefficients were fit on (already within bounds by
    construction) and acts as a safety net when applying fixed coefficients to out-of-sample x, e.g.
    a bootstrap resample.
    """
    raw = pd.DataFrame(x.to_numpy()[:, None] * b.to_numpy()[None, :] + a.to_numpy()[None, :], index=x.index, columns=a.index)
    clipped = raw.clip(lower=0.0, upper=1.0)
    return clipped.div(clipped.sum(axis=1), axis=0)


def linear_portfolio_returns(returns: pd.DataFrame, a: pd.Series, b: pd.Series, x: pd.Series) -> pd.Series:
    """Portfolio returns from applying the linear weight function w_i(t) = a_i + b_i*x(t)."""
    weights = linear_weights_matrix(a, b, x)
    return (returns * weights).sum(axis=1)


def max_drawdown(cumulative: pd.Series) -> float:
    """Maximum drawdown of a cumulative price/growth series."""
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return float(drawdown.min())


def performance_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized return, volatility, Sharpe ratio and max drawdown for each column."""
    n_months = len(returns)
    cagr = (1.0 + returns).prod() ** (MONTHS_PER_YEAR / n_months) - 1.0
    ann_vol = returns.std() * np.sqrt(MONTHS_PER_YEAR)
    mean_ann_return = returns.mean() * MONTHS_PER_YEAR
    sharpe = (mean_ann_return - RISK_FREE_RATE) / ann_vol
    cumulative = (1.0 + returns).cumprod()
    mdd = cumulative.apply(max_drawdown)

    summary = pd.DataFrame(
        {
            "Performance (CAGR)": cagr,
            "Volatility (ann.)": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": mdd,
        }
    )
    return summary


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Format a performance summary DataFrame as percentages / rounded ratio for printing."""
    formatted = summary.copy()
    for col in ["Performance (CAGR)", "Volatility (ann.)", "Max Drawdown"]:
        formatted[col] = formatted[col].map(lambda v: f"{v:.2%}")
    formatted["Sharpe Ratio"] = formatted["Sharpe Ratio"].map(lambda v: f"{v:.2f}")
    return formatted


def analyze_and_report(returns: pd.DataFrame, title: str) -> tuple[pd.Series, pd.Series]:
    """Optimize the Sharpe-maximizing portfolio for the given returns, print its report.

    Returns the (weights, portfolio_returns) so callers can reuse the weights elsewhere.
    """
    weights = optimize_sharpe(returns)
    portfolio_returns = returns @ weights
    portfolio_returns.name = "Optimal Portfolio"

    all_returns = returns.copy()
    all_returns[portfolio_returns.name] = portfolio_returns
    summary = performance_summary(all_returns)

    print(f"\n=== {title} ({len(returns)} months) ===")
    print("Optimal weights (max Sharpe):")
    print(weights.map(lambda w: f"{w:.2%}").to_string())
    print()
    print("Performance summary:")
    print(format_summary(summary).to_string())

    return weights, portfolio_returns


def analyze_and_report_linear(returns: pd.DataFrame, x: pd.Series, title: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Fit linear weights w_i(t) = a_i + b_i*x(t) maximizing Sharpe, print coefficients and performance.

    Returns (intercepts, slopes, portfolio_returns) so callers can reuse the fit elsewhere.
    """
    a, b = optimize_linear_weights(returns, x)
    portfolio_returns = linear_portfolio_returns(returns, a, b, x)
    portfolio_returns.name = "Optimal Portfolio"

    all_returns = returns.copy()
    all_returns[portfolio_returns.name] = portfolio_returns
    summary = performance_summary(all_returns)

    print(f"\n=== {title} ({len(returns)} months) ===")
    print("Linear weight coefficients (Weight = Intercept + Slope * x):")
    print(pd.DataFrame({"Intercept": a, "Slope": b}).map(lambda v: f"{v:.4f}").to_string())
    print()
    print("Performance summary:")
    print(format_summary(summary).to_string())

    return a, b, portfolio_returns


def format_interval(interval: pd.Interval) -> str:
    """Format a pandas Interval of excess CAPE yield (%) values for display, handling infinite edges."""
    lo = "-inf" if interval.left == -np.inf else f"{interval.left:.2f}%"
    hi = "+inf" if interval.right == np.inf else f"{interval.right:.2f}%"
    return f"{lo} to {hi}"


def quantile_buckets(ecy: pd.Series, quantiles=DEFAULT_QUANTILES) -> list[tuple[str, pd.Series]]:
    """Split months into ranges bounded by the given quantiles of the excess CAPE yield.

    Returns a list of (label, boolean mask aligned with ecy.index) tuples, ordered from
    lowest to highest excess CAPE yield.
    """
    thresholds = ecy.quantile(list(quantiles)).tolist()
    edges = [-np.inf] + thresholds + [np.inf]
    cuts = pd.cut(ecy, bins=edges, include_lowest=True)
    return [(format_interval(interval), cuts == interval) for interval in cuts.cat.categories]


def regime_switching_weights_and_returns(
    returns: pd.DataFrame, ecy: pd.Series, quantiles
) -> tuple[list[pd.Series], pd.Series]:
    """Optimize each Excess-CAPE-Yield bucket's weights and build the Regime-Switching Portfolio's return series."""
    bucket_weights = []
    parts = []
    for _, mask in quantile_buckets(ecy, quantiles):
        bucket_returns = returns.loc[mask]
        weights = optimize_sharpe(bucket_returns)
        bucket_weights.append(weights)
        parts.append(bucket_returns @ weights)
    portfolio_returns = pd.concat(parts).sort_index()
    portfolio_returns.name = "Regime-Switching Portfolio"
    return bucket_weights, portfolio_returns


def circular_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a circular block bootstrap resample of positions 0..n-1, wrapping past the end."""
    n_blocks = -(-n // block_size)  # ceil division
    starts = rng.integers(0, n, size=n_blocks)
    blocks = (starts[:, None] + np.arange(block_size)) % n
    return blocks.reshape(-1)[:n]


def block_bootstrap_resample(
    returns: pd.DataFrame, ecy: pd.Series, block_size: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample returns and excess CAPE yield with the same circular block indices, position-indexed."""
    idx = circular_block_indices(len(returns), block_size, rng)
    resampled_returns = returns.iloc[idx].reset_index(drop=True)
    resampled_ecy = ecy.iloc[idx].reset_index(drop=True)
    return resampled_returns, resampled_ecy


def evaluate_regime_switching(
    resampled_returns: pd.DataFrame, resampled_ecy: pd.Series, quantiles, bucket_weights: list[pd.Series]
) -> pd.Series:
    """Apply already-optimized per-bucket weights (lowest to highest ECY) to a return series, assigning
    months to buckets by re-deriving Excess CAPE Yield quantile thresholds from resampled_ecy."""
    parts = [
        resampled_returns.loc[mask] @ weights
        for (_, mask), weights in zip(quantile_buckets(resampled_ecy, quantiles), bucket_weights)
    ]
    portfolio_returns = pd.concat(parts).sort_index()
    portfolio_returns.name = "Regime-Switching Portfolio"
    return portfolio_returns


def run_block_bootstrap(
    returns: pd.DataFrame,
    ecy: pd.Series,
    full_weights: pd.Series,
    bucket_weights: list[pd.Series],
    ecy_value_coeffs: tuple[pd.Series, pd.Series],
    ecy_percentile_coeffs: tuple[pd.Series, pd.Series],
    quantiles=DEFAULT_QUANTILES,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
) -> dict:
    """Run a circular block bootstrap that evaluates the already-optimized Full-Period,
    Regime-Switching and linear-weight (in ECY value / ECY percentile) portfolios (all fixed
    weights/coefficients) against resampled return paths. Only the Excess-CAPE-Yield bucket
    membership (Regime-Switching) and percentile rank (linear-in-percentile) are re-derived per
    resample, not the weights/coefficients themselves.

    Returns a dict with "<key>_metrics" (one row per bootstrap iteration) and "<key>_growth"
    (one resampled cumulative growth path per row, same column order as `returns`) for each of
    "full", "regime", "ecy_value" and "ecy_percentile".
    """
    rng = np.random.default_rng()
    ecy_value_a, ecy_value_b = ecy_value_coeffs
    ecy_percentile_a, ecy_percentile_b = ecy_percentile_coeffs

    rows: dict[str, list] = {
        "full_metrics": [], "full_growth": [],
        "regime_metrics": [], "regime_growth": [],
        "ecy_value_metrics": [], "ecy_value_growth": [],
        "ecy_percentile_metrics": [], "ecy_percentile_growth": [],
    }

    for _ in range(n_iterations):
        resampled_returns, resampled_ecy = block_bootstrap_resample(returns, ecy, block_size, rng)

        full_portfolio_returns = resampled_returns @ full_weights
        rows["full_metrics"].append(performance_summary(full_portfolio_returns.to_frame(name="Optimal Portfolio")).iloc[0])
        rows["full_growth"].append((1.0 + full_portfolio_returns).cumprod().to_numpy())

        regime_returns = evaluate_regime_switching(resampled_returns, resampled_ecy, quantiles, bucket_weights)
        rows["regime_metrics"].append(performance_summary(regime_returns.to_frame()).iloc[0])
        rows["regime_growth"].append((1.0 + regime_returns).cumprod().to_numpy())

        ecy_value_returns = linear_portfolio_returns(resampled_returns, ecy_value_a, ecy_value_b, resampled_ecy)
        rows["ecy_value_metrics"].append(performance_summary(ecy_value_returns.to_frame(name="Optimal Portfolio")).iloc[0])
        rows["ecy_value_growth"].append((1.0 + ecy_value_returns).cumprod().to_numpy())

        resampled_percentile = resampled_ecy.rank(pct=True)
        ecy_percentile_returns = linear_portfolio_returns(resampled_returns, ecy_percentile_a, ecy_percentile_b, resampled_percentile)
        rows["ecy_percentile_metrics"].append(performance_summary(ecy_percentile_returns.to_frame(name="Optimal Portfolio")).iloc[0])
        rows["ecy_percentile_growth"].append((1.0 + ecy_percentile_returns).cumprod().to_numpy())

    return {
        "full_metrics": pd.DataFrame(rows["full_metrics"]),
        "full_growth": np.array(rows["full_growth"]),
        "regime_metrics": pd.DataFrame(rows["regime_metrics"]),
        "regime_growth": np.array(rows["regime_growth"]),
        "ecy_value_metrics": pd.DataFrame(rows["ecy_value_metrics"]),
        "ecy_value_growth": np.array(rows["ecy_value_growth"]),
        "ecy_percentile_metrics": pd.DataFrame(rows["ecy_percentile_metrics"]),
        "ecy_percentile_growth": np.array(rows["ecy_percentile_growth"]),
    }


def growth_ci_band(
    growth_paths: np.ndarray, index: pd.Index, ci_level: float = BOOTSTRAP_CI_LEVEL
) -> tuple[pd.Series, pd.Series]:
    """Per-period percentile band (lower, upper) of raw cumulative growth across bootstrap iterations."""
    alpha = (1.0 - ci_level) / 2.0
    lower = pd.Series(np.quantile(growth_paths, alpha, axis=0), index=index)
    upper = pd.Series(np.quantile(growth_paths, 1.0 - alpha, axis=0), index=index)
    return lower, upper


def summarize_bootstrap_ci(
    original: pd.Series, samples: pd.DataFrame, ci_level: float = BOOTSTRAP_CI_LEVEL
) -> pd.DataFrame:
    """Combine point estimates (from the original, non-resampled data) with bootstrap percentile CIs."""
    alpha = (1.0 - ci_level) / 2.0
    lower = samples.quantile(alpha)
    upper = samples.quantile(1.0 - alpha)
    return pd.DataFrame({"Estimate": original, "CI Lower": lower, "CI Upper": upper})


def format_ci_table(ci: pd.DataFrame) -> pd.DataFrame:
    """Format a bootstrap CI table as percentages, except rows named "Sharpe Ratio" shown as a ratio."""
    formatted = ci.copy()
    for col in formatted.columns:
        formatted[col] = [
            f"{v:.2f}" if idx == "Sharpe Ratio" else f"{v:.2%}" for idx, v in zip(formatted.index, formatted[col])
        ]
    return formatted


def format_quantile_scan(scan: pd.DataFrame) -> pd.DataFrame:
    """Format a quantile-scan DataFrame with MultiIndex (metric, stat) columns for printing."""
    formatted = scan.copy()
    for metric in formatted.columns.get_level_values(0).unique():
        fmt = (lambda v: f"{v:.2f}") if metric == "Sharpe Ratio" else (lambda v: f"{v:.2%}")
        for stat in formatted[metric].columns:
            formatted[(metric, stat)] = formatted[(metric, stat)].map(fmt)
    return formatted


def scan_quantile_counts(
    returns: pd.DataFrame,
    ecy: pd.Series,
    max_quantiles: int = 9,
    n_iterations: int = BOOTSTRAP_SCAN_ITERATIONS,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> pd.DataFrame:
    """Scan 1 to max_quantiles equidistant quantile cut points (2 to max_quantiles+1 buckets).

    Returns a DataFrame indexed by number of quantiles, with MultiIndex columns (metric, stat)
    where stat is one of "Estimate", "CI Lower", "CI Upper". For each quantile count, weights are
    optimized once on the original data and then evaluated (not re-optimized) against each resample.
    """
    rng = np.random.default_rng()
    rows = {}
    for n_quantiles in range(1, max_quantiles + 1):
        quantiles = tuple(i / (n_quantiles + 1) for i in range(1, n_quantiles + 1))
        bucket_weights, portfolio_returns = regime_switching_weights_and_returns(returns, ecy, quantiles)
        estimate = performance_summary(portfolio_returns.to_frame()).iloc[0]

        metrics_rows = []
        for _ in range(n_iterations):
            resampled_returns, resampled_ecy = block_bootstrap_resample(returns, ecy, block_size, rng)
            resampled_portfolio_returns = evaluate_regime_switching(resampled_returns, resampled_ecy, quantiles, bucket_weights)
            metrics_rows.append(performance_summary(resampled_portfolio_returns.to_frame()).iloc[0])
        ci = summarize_bootstrap_ci(estimate, pd.DataFrame(metrics_rows), ci_level)
        rows[n_quantiles] = ci.stack()
    scan = pd.DataFrame(rows).T
    scan.index.name = "Quantiles"
    return scan


def plot_quantile_scan(scan: pd.DataFrame, ci_level: float = BOOTSTRAP_CI_LEVEL) -> None:
    """Plot the Regime-Switching Portfolio's Sharpe ratio, with its bootstrap CI band, vs quantile count."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        scan.index, scan[("Sharpe Ratio", "CI Lower")], scan[("Sharpe Ratio", "CI Upper")],
        alpha=0.2, label=f"{ci_level:.0%} bootstrap CI",
    )
    ax.plot(scan.index, scan[("Sharpe Ratio", "Estimate")], marker="o", label="Sharpe Ratio")
    ax.set_xlabel("Number of Equidistant Quantiles")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Regime-Switching Portfolio Sharpe Ratio vs Number of Quantiles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_linear_weights(
    a: pd.Series, b: pd.Series, x: pd.Series, xlabel: str, title: str,
    x_axis_map: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Plot each asset's linear weight function w_i(x) = a_i + b_i*x (clipped/renormalized) over the observed range of x.

    x_axis_map, if given, maps the linspace grid of x (e.g. an Excess CAPE Yield percentile) to the
    values shown on the x-axis (e.g. actual Excess CAPE Yield via its empirical quantile function),
    producing a nonlinear curve when displayed against a variable other than the one w is linear in.
    """
    x_grid = np.linspace(x.min(), x.max(), 200)
    weights = linear_weights_matrix(a, b, pd.Series(x_grid))
    x_axis = x_axis_map(x_grid) if x_axis_map is not None else x_grid

    fig, ax = plt.subplots(figsize=(12, 7))
    for col in weights.columns:
        ax.plot(x_axis, weights[col], label=col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_normalized(
    prices: pd.DataFrame,
    portfolio_growth: pd.Series,
    title: str = "Normalized Performance (start = 100)",
    ci_band: tuple[pd.Series, pd.Series] | None = None,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
) -> None:
    """Plot constituents and a portfolio growth series, all normalized to start at 100.

    ci_band, if given, is a (lower, upper) pair of raw cumulative growth bootstrap percentiles
    (same baseline/scale as portfolio_growth) shaded around the portfolio line.
    """
    normalized = prices / prices.iloc[0] * 100.0
    portfolio_normalized = portfolio_growth / portfolio_growth.iloc[0] * 100.0
    label = portfolio_growth.name or "Optimal Portfolio"

    fig, ax = plt.subplots(figsize=(14, 8))
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col, alpha=0.7)
    if ci_band is not None:
        lower, upper = ci_band
        lower_normalized = lower / portfolio_growth.iloc[0] * 100.0
        upper_normalized = upper / portfolio_growth.iloc[0] * 100.0
        ax.fill_between(
            portfolio_normalized.index, lower_normalized, upper_normalized,
            color="black", alpha=0.15, label=f"{label} {ci_level:.0%} bootstrap CI",
        )
    ax.plot(portfolio_normalized.index, portfolio_normalized, label=label, color="black", linewidth=2.5)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Value")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    prices = load_all_prices()
    returns = prices.pct_change().dropna(how="any")

    ecy = load_excess_cape_yield()
    ecy.index = ecy.index.to_timestamp(how="end").normalize()

    # Align the excess CAPE yield with the existing series' common date range.
    common_dates = returns.index.intersection(ecy.index)
    returns = returns.loc[common_dates]
    ecy = ecy.loc[common_dates]

    # Keep the price row immediately preceding the first aligned return as the plot baseline.
    start_pos = max(prices.index.get_indexer([common_dates.min()])[0] - 1, 0)
    prices = prices.iloc[start_pos:]

    print(f"Common date range: {prices.index.min().date()} to {prices.index.max().date()} "
          f"({len(returns)} months)")
    print(f"Latest Excess CAPE Yield: {ecy.iloc[-1]:.2f}%\n")

    full_weights, portfolio_returns = analyze_and_report(returns, "Full Period")
    full_metrics = performance_summary(portfolio_returns.to_frame()).iloc[0]
    portfolio_growth = (1.0 + portfolio_returns).cumprod()

    print(f"\n\nExcess CAPE Yield quantiles ({DEFAULT_QUANTILES}): "
          f"{[f'{q:.2f}%' for q in ecy.quantile(list(DEFAULT_QUANTILES))]}")
    regime_returns_parts = []
    bucket_weights_list = []
    for label, mask in quantile_buckets(ecy, DEFAULT_QUANTILES):
        bucket_weights, bucket_portfolio_returns = analyze_and_report(returns.loc[mask], f"Excess CAPE Yield {label}")
        regime_returns_parts.append(bucket_portfolio_returns)
        bucket_weights_list.append(bucket_weights)

    # Chronological returns from rebalancing into each month's excess-CAPE-yield-range optimal weights.
    regime_returns = pd.concat(regime_returns_parts).sort_index()
    regime_returns.name = "Regime-Switching Portfolio"

    all_returns_with_regime = returns.copy()
    all_returns_with_regime[regime_returns.name] = regime_returns
    regime_summary = performance_summary(all_returns_with_regime)
    regime_metrics = regime_summary.loc[regime_returns.name]

    print(f"\n=== Regime-Switching Portfolio (rebalanced across Excess CAPE Yield ranges, {len(regime_returns)} months) ===")
    print("Performance summary:")
    print(format_summary(regime_summary).to_string())

    regime_growth = (1.0 + regime_returns).cumprod()

    ecy_percentile = ecy.rank(pct=True)
    ecy_value_a, ecy_value_b, ecy_value_returns = analyze_and_report_linear(
        returns, ecy, "Linear Weights (Excess CAPE Yield)")
    ecy_value_returns.name = "Linear Weights (ECY)"
    ecy_value_metrics = performance_summary(ecy_value_returns.to_frame()).iloc[0]
    ecy_value_growth = (1.0 + ecy_value_returns).cumprod()
    plot_linear_weights(
        ecy_value_a, ecy_value_b, ecy,
        xlabel="Excess CAPE Yield (%)", title="Portfolio Weights vs Excess CAPE Yield",
    )

    ecy_percentile_a, ecy_percentile_b, ecy_percentile_returns = analyze_and_report_linear(
        returns, ecy_percentile, "Linear Weights (Excess CAPE Yield Percentile)")
    ecy_percentile_returns.name = "Linear Weights (ECY Percentile)"
    ecy_percentile_metrics = performance_summary(ecy_percentile_returns.to_frame()).iloc[0]
    ecy_percentile_growth = (1.0 + ecy_percentile_returns).cumprod()
    plot_linear_weights(
        ecy_percentile_a, ecy_percentile_b, ecy_percentile,
        xlabel="Excess CAPE Yield (%)", title="Portfolio Weights vs Excess CAPE Yield Percentile",
        x_axis_map=lambda grid: ecy.quantile(grid).to_numpy(),
    )

    print(f"\n=== Block Bootstrap Confidence Intervals "
          f"(block size={BOOTSTRAP_BLOCK_SIZE} months, {BOOTSTRAP_ITERATIONS} iterations, "
          f"{BOOTSTRAP_CI_LEVEL:.0%} CI) ===")
    bootstrap = run_block_bootstrap(
        returns, ecy, full_weights, bucket_weights_list,
        (ecy_value_a, ecy_value_b), (ecy_percentile_a, ecy_percentile_b), DEFAULT_QUANTILES,
    )

    full_growth_ci = growth_ci_band(bootstrap["full_growth"], returns.index)
    plot_normalized(prices, portfolio_growth, ci_band=full_growth_ci)

    regime_growth_ci = growth_ci_band(bootstrap["regime_growth"], returns.index)
    plot_normalized(
        prices, regime_growth, title="Normalized Performance \u2014 Regime-Switching Portfolio (start = 100)",
        ci_band=regime_growth_ci,
    )

    ecy_value_growth_ci = growth_ci_band(bootstrap["ecy_value_growth"], returns.index)
    plot_normalized(
        prices, ecy_value_growth, title="Normalized Performance \u2014 Linear Weights (Excess CAPE Yield, start = 100)",
        ci_band=ecy_value_growth_ci,
    )

    ecy_percentile_growth_ci = growth_ci_band(bootstrap["ecy_percentile_growth"], returns.index)
    plot_normalized(
        prices, ecy_percentile_growth,
        title="Normalized Performance \u2014 Linear Weights (Excess CAPE Yield Percentile, start = 100)",
        ci_band=ecy_percentile_growth_ci,
    )

    print("\nFull-Period Optimal Portfolio performance:")
    print(format_ci_table(summarize_bootstrap_ci(full_metrics, bootstrap["full_metrics"])).to_string())

    print("\nRegime-Switching Portfolio performance:")
    print(format_ci_table(summarize_bootstrap_ci(regime_metrics, bootstrap["regime_metrics"])).to_string())

    print("\nLinear Weights (Excess CAPE Yield) performance:")
    print(format_ci_table(summarize_bootstrap_ci(ecy_value_metrics, bootstrap["ecy_value_metrics"])).to_string())

    print("\nLinear Weights (Excess CAPE Yield Percentile) performance:")
    print(format_ci_table(summarize_bootstrap_ci(ecy_percentile_metrics, bootstrap["ecy_percentile_metrics"])).to_string())

    scan = scan_quantile_counts(returns, ecy, max_quantiles=5)
    print("\n=== Regime-Switching Portfolio: scanning 1-5 equidistant quantiles ===")
    print(format_quantile_scan(scan).to_string())
    plot_quantile_scan(scan)


if __name__ == "__main__":
    main()
