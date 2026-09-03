"""Portfolio optimizer: maximize the Sharpe ratio across MSCI World factor indices, gold and silver.

Loads historical monthly prices for:
  - MSCI World, MSCI World Value, MSCI World Momentum, MSCI World Energy, MSCI World High Dividend Yield (xlsx)
  - Gold spot price, Silver spot price (csv)

Aligns all series to their maximum common date range, finds the long-only weights that
maximize the Sharpe ratio (mean-variance optimization), and reports performance,
volatility, Sharpe ratio and maximum drawdown for the optimal portfolio and its constituents.
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = Path(__file__).parent / "data"

# Annualized risk-free rate assumption used in the Sharpe ratio calculation.
RISK_FREE_RATE = 0.0

MONTHS_PER_YEAR = 12

# Quantiles of the excess CAPE yield used to split history into valuation regimes.
# DEFAULT_QUANTILES = [1/2]
DEFAULT_QUANTILES = (1/3, 2/3)

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


def load_metal_price(path: Path, name: str) -> pd.Series:
    """Load a monthly metal spot price csv (MM/YYYY dates) and return a Series indexed by month Period."""
    df = pd.read_csv(path)
    price_col = df.columns[1]
    period = pd.to_datetime(df["Date"], format="%m/%Y").dt.to_period("M")
    series = pd.Series(df[price_col].values, index=period, name=name)
    return series


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
    series = [load_msci_index(p) for p in sorted(glob.glob(str(DATA_DIR / "*.xlsx")))]
    series.append(load_metal_price(DATA_DIR / "gold.csv", "Gold"))
    series.append(load_metal_price(DATA_DIR / "silver.csv", "Silver"))

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
    """Like regime_switching_returns, but also returns each bucket's optimal weights (lowest to highest ECY)."""
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


def regime_switching_returns(returns: pd.DataFrame, ecy: pd.Series, quantiles) -> pd.Series:
    """Build the Regime-Switching Portfolio's return series, optimizing each bucket's own weights quietly."""
    _, portfolio_returns = regime_switching_weights_and_returns(returns, ecy, quantiles)
    return portfolio_returns


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


def run_block_bootstrap(
    returns: pd.DataFrame,
    ecy: pd.Series,
    quantiles=DEFAULT_QUANTILES,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
) -> dict:
    """Run a nested circular block bootstrap: re-optimize the Full-Period portfolio and, within each
    resample, re-derive the Excess-CAPE-Yield buckets and re-optimize the Regime-Switching portfolio.

    Returns a dict with "full_weights", "full_metrics", "bucket_weights" (list of per-bucket weight
    DataFrames, lowest to highest ECY) and "regime_metrics", each a DataFrame with one row per
    successful bootstrap iteration, plus "full_growth" and "regime_growth" arrays (one resampled
    cumulative growth path per row, same column order as `returns`) used to band the growth plots.
    """
    rng = np.random.default_rng()
    n_buckets = len(quantiles) + 1

    full_weights_rows = []
    full_metrics_rows = []
    full_growth_rows = []
    bucket_weights_rows: list[list[pd.Series]] = [[] for _ in range(n_buckets)]
    regime_metrics_rows = []
    regime_growth_rows = []
    n_failed = 0

    for _ in range(n_iterations):
        try:
            resampled_returns, resampled_ecy = block_bootstrap_resample(returns, ecy, block_size, rng)

            full_weights = optimize_sharpe(resampled_returns)
            full_portfolio_returns = resampled_returns @ full_weights
            full_metrics = performance_summary(full_portfolio_returns.to_frame(name="Optimal Portfolio")).iloc[0]

            bucket_weights, regime_returns = regime_switching_weights_and_returns(
                resampled_returns, resampled_ecy, quantiles
            )
            if len(bucket_weights) != n_buckets:
                raise ValueError("Resample produced an unexpected number of Excess CAPE Yield buckets")
            regime_metrics = performance_summary(regime_returns.to_frame()).iloc[0]
        except (RuntimeError, ValueError):
            n_failed += 1
            continue

        full_weights_rows.append(full_weights)
        full_metrics_rows.append(full_metrics)
        full_growth_rows.append((1.0 + full_portfolio_returns).cumprod().to_numpy())
        for bucket_rows, weights in zip(bucket_weights_rows, bucket_weights):
            bucket_rows.append(weights)
        regime_metrics_rows.append(regime_metrics)
        regime_growth_rows.append((1.0 + regime_returns).cumprod().to_numpy())

    if n_failed:
        print(f"Warning: {n_failed}/{n_iterations} bootstrap iterations failed to converge and were skipped.")

    return {
        "full_weights": pd.DataFrame(full_weights_rows),
        "full_metrics": pd.DataFrame(full_metrics_rows),
        "full_growth": np.array(full_growth_rows),
        "bucket_weights": [pd.DataFrame(rows) for rows in bucket_weights_rows],
        "regime_metrics": pd.DataFrame(regime_metrics_rows),
        "regime_growth": np.array(regime_growth_rows),
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
    where stat is one of "Estimate", "CI Lower", "CI Upper" from a block bootstrap re-run at
    each quantile count (each resample re-derives its own Excess CAPE Yield buckets).
    """
    rng = np.random.default_rng()
    rows = {}
    for n_quantiles in range(1, max_quantiles + 1):
        quantiles = tuple(i / (n_quantiles + 1) for i in range(1, n_quantiles + 1))
        portfolio_returns = regime_switching_returns(returns, ecy, quantiles)
        estimate = performance_summary(portfolio_returns.to_frame()).iloc[0]

        metrics_rows = []
        for _ in range(n_iterations):
            resampled_returns, resampled_ecy = block_bootstrap_resample(returns, ecy, block_size, rng)
            try:
                resampled_portfolio_returns = regime_switching_returns(resampled_returns, resampled_ecy, quantiles)
            except RuntimeError:
                continue
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
    bucket_labels = []
    bucket_weights_list = []
    for label, mask in quantile_buckets(ecy, DEFAULT_QUANTILES):
        bucket_weights, bucket_portfolio_returns = analyze_and_report(returns.loc[mask], f"Excess CAPE Yield {label}")
        regime_returns_parts.append(bucket_portfolio_returns)
        bucket_labels.append(label)
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

    print(f"\n=== Block Bootstrap Confidence Intervals "
          f"(block size={BOOTSTRAP_BLOCK_SIZE} months, {BOOTSTRAP_ITERATIONS} iterations, "
          f"{BOOTSTRAP_CI_LEVEL:.0%} CI) ===")
    bootstrap = run_block_bootstrap(returns, ecy, DEFAULT_QUANTILES)

    full_growth_ci = growth_ci_band(bootstrap["full_growth"], returns.index)
    plot_normalized(prices, portfolio_growth, ci_band=full_growth_ci)

    regime_growth_ci = growth_ci_band(bootstrap["regime_growth"], returns.index)
    plot_normalized(
        prices, regime_growth, title="Normalized Performance \u2014 Regime-Switching Portfolio (start = 100)",
        ci_band=regime_growth_ci,
    )

    print("\nFull-Period Optimal Portfolio weights:")
    print(format_ci_table(summarize_bootstrap_ci(full_weights, bootstrap["full_weights"])).to_string())
    print("\nFull-Period Optimal Portfolio performance:")
    print(format_ci_table(summarize_bootstrap_ci(full_metrics, bootstrap["full_metrics"])).to_string())

    for label, original_bucket_weights, bucket_samples in zip(bucket_labels, bucket_weights_list, bootstrap["bucket_weights"]):
        print(f"\nRegime-Switching bucket [{label}] weights:")
        print(format_ci_table(summarize_bootstrap_ci(original_bucket_weights, bucket_samples)).to_string())

    print("\nRegime-Switching Portfolio performance:")
    print(format_ci_table(summarize_bootstrap_ci(regime_metrics, bootstrap["regime_metrics"])).to_string())

    scan = scan_quantile_counts(returns, ecy, max_quantiles=5)
    print("\n=== Regime-Switching Portfolio: scanning 1-5 equidistant quantiles ===")
    print(format_quantile_scan(scan).to_string())
    plot_quantile_scan(scan)


if __name__ == "__main__":
    main()
