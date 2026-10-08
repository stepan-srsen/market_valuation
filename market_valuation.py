import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data import (
    add_bands,
    detect_bear_markets,
    fetch_fred_csv,
    fetch_sp500_earnings,
    fetch_yfinance,
    fit_exponential,
    get_CPI_scaling,
    get_cpi,
    get_gdp,
    get_inflation,
)

# TODO: implement Price to Sales ratio metric
# TODO: implement earnings yield gap metric
# TODO: ?implement exponential moving average for smoothing inflation
# TODO: ?add seasonal trends to CPI prediction from historical seasonally adjusted vs non-adjusted data

# Financial crises dictionary with approximate peak/trough dates
FINANCIAL_CRISES = {
    "Great Depression": dt.datetime(1929, 9, 16),
    "Kennedy Slide of 1962": dt.datetime(1961, 12, 12),
    "Vietnam war, inflation, FED rates": dt.datetime(1968, 11, 29),
    "Bretton Woods System end, Oil Crisis": dt.datetime(1973, 1, 11),
    "Inflation, FED Rates, Oil Crisis": dt.datetime(1980, 11, 28),
    "Black Monday": dt.datetime(1987, 10, 19),
    "Dot-com Bubble": dt.datetime(2000, 3, 10),
    "9/11 Attacks": dt.datetime(2001, 9, 11),
    "Global Financial Crisis": dt.datetime(2007, 10, 9),
    "Flash Crash": dt.datetime(2010, 5, 6),
    "Euro Debt Crisis, Ratings": dt.datetime(2011, 8, 1),
    "China Slowdown": dt.datetime(2015, 8, 18),
    "Crude Oil Falling": dt.datetime(2016, 1, 20),
    "Cryptocurrency Crash": dt.datetime(2018, 9, 20),
    "COVID-19": dt.datetime(2020, 2, 20),
    "Inflation, FED Rates, Invasion": dt.datetime(2022, 1, 3),
    "Trade War": dt.datetime(2025, 4, 3),
}

def calc_cape_ratio(averaging_years: int = 10) -> pd.Series:
    """Calculate (Shiller's-like) CAPE (Cyclically Adjusted Price-to-Earnings) ratio.
    
    Args:
        averaging_years: Number of years to average earnings over (default: 10 years)

    Returns:
        Time series of CAPE ratio values
    """
    # Fetch S&P 500 prices
    prices = fetch_yfinance('^GSPC', auto_adjust=False).resample('D').ffill()

    # Fetch S&P 500 earnings
    earnings = fetch_sp500_earnings()
    # Interpolate to get smooth evolution
    earnings = earnings.resample('D').interpolate(method='linear')
    
    # Fetch CPI data
    cpi = get_cpi()
    
    # Align all series to the same index
    earnings = earnings.reindex(prices.index, method='ffill')
    cpi = cpi.reindex(prices.index, method='ffill')

    # Adjust for inflation (normalize to latest CPI value)
    latest_cpi = cpi.iloc[-1]
    real_prices = prices * (latest_cpi / cpi)
    real_earnings = earnings * (latest_cpi / cpi)

    # Calculate rolling average of real earnings
    avg_real_earnings = real_earnings.rolling(window=int(averaging_years * 365), min_periods=int(averaging_years * 365)).mean()
    
    # Calculate CAPE ratio
    cape_ratio = real_prices / avg_real_earnings
    cape_ratio.name = f"CAPE Ratio ({averaging_years}yr)"
    
    return cape_ratio.dropna()

def calc_treasury_cape_ratio(averaging_years: int = 10) -> pd.Series:
    """Calculate 10Y Treasury to CAPE yield ratio.
    
    Args:
        averaging_years: Number of years to average earnings over (default: 10 years)
    """
    cape = calc_cape_ratio(averaging_years=averaging_years).resample('D').ffill()
    ti10y = fetch_fred_csv("DGS10")
    # Align all series to the same index
    ti10y = ti10y.reindex(cape.index, method='ffill')
    # Calculate CAPE to Treasury yield ratio and drop NA values
    cape_ti10y_ratio = (cape * ti10y / 100.0).dropna()
    return cape_ti10y_ratio.rename(f"10Y Treasury Yield / CAPE Yield ({averaging_years}yr) Ratio")

def calc_excess_cape_yield(averaging_years: int = 10) -> pd.Series:
    """Calculate excess CAPE yield (CAPE earnings yield - 10Y Treasury yield).
    
    Args:
        averaging_years: Number of years to average earnings over (default: 10 years)
    """
    cape = calc_cape_ratio(averaging_years=averaging_years).resample('D').ffill()
    ti10y = fetch_fred_csv("DGS10")
    inflation = get_inflation(averaging_years=averaging_years)
    # Align all series to the same index
    ti10y = ti10y.reindex(cape.index, method='ffill')
    inflation = inflation.reindex(cape.index, method='ffill')
    # Calculate excess CAPE yield and drop NA values
    excess_cape_yield = (1.0/cape - ti10y/100.0 + inflation).dropna()
    return excess_cape_yield.rename(f"Excess CAPE Yield ({averaging_years}yr)")

def calc_buffett_indicator() -> pd.Series:
    """Calculate the Buffett Indicator (Market Cap / GDP ratio).
    
    Args:
        exponential_fit: If True, fit an exponential trend and return the ratio
                        divided by the exponential fit (detrended ratio).
        detrend: If True and exponential_fit is True, return detrended data
    """
    # Fetch market cap data
    market_cap = fetch_yfinance('^W5000').resample('D').ffill() # '^W5000' vs '^FTW5000' ticker
    # Fetch GDP data with GDPNow extension
    gdp_data = get_gdp()
    # Align GDP to market cap dates
    gdp_aligned = gdp_data.reindex(market_cap.index, method='ffill')
    # Calculate ratio
    ratio = market_cap / gdp_aligned
    ratio.name = "Buffett Indicator"
    
    return ratio.dropna()

def plot_dual_axis(left_datasets, right_datasets=[], bear_markets=None, x_axis_labels=None, normalize_left=False, normalize_right=False, plot_from="max") -> None:
    """Plot datasets on dual y-axes with optional normalization.
    
    Args:
        left_datasets: Single Series/DataFrame or list of Series/DataFrames for left axis
        right_datasets: Single Series/DataFrame or list of Series/DataFrames for right axis
        bear_markets: List of (start, end) tuples for bear market periods
        x_axis_labels: Dictionary of {label: date} for significant events on x-axis
        normalize_left: If True, normalize left axis datasets to their maximum
        normalize_right: If True, normalize right axis datasets to their maximum
        plot_from: Plot start date, either a predefined mode ("min"/"max": min/max of all
               datasets' minima, "min_left"/"max_left": min/max of left datasets' minima,
               "min_right"/"max_right": min/max of right datasets' minima),
               or a specific date (datetime, Timestamp, or date string)
    """
    # Predefined values for the plot_from argument (plot start date selection)
    start_modes = {
        "min": ("min", "all"),
        "max": ("max", "all"),
        "min_left": ("min", "left"),
        "max_left": ("max", "left"),
        "min_right": ("min", "right"),
        "max_right": ("max", "right"),
    }
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    # Convert single datasets to lists
    left_datasets = left_datasets if isinstance(left_datasets, list) else [left_datasets]
    right_datasets = right_datasets if isinstance(right_datasets, list) else [right_datasets]

    # Determine the plot start date
    if isinstance(plot_from, str) and plot_from in start_modes:
        aggregate, scope = start_modes[plot_from]
        sources = {"all": left_datasets + right_datasets, "left": left_datasets, "right": right_datasets}[scope]
        if not sources:
            raise ValueError(f"No datasets available to resolve plot_from={plot_from!r}.")
        minima = [ds.index.min() for ds in sources]
        plot_start = max(minima) if aggregate == "max" else min(minima)
    elif isinstance(plot_from, (dt.datetime, pd.Timestamp)):
        plot_start = pd.Timestamp(plot_from)
    elif isinstance(plot_from, str):
        # Assume any other string is a parseable date
        plot_start = pd.Timestamp(plot_from)
    else:
        raise ValueError(f"Invalid plot_from value {plot_from!r}: expected one of {sorted(start_modes)} or a date.")
    plot_end = max(ds.index.max() for ds in left_datasets + right_datasets)
    
    # Normalize function
    def normalize_dataset(data):
        """Normalize dataset by maximum of first column/series."""
        if isinstance(data, pd.DataFrame):
            # For DataFrame, divide all columns by the maximum of the first column
            first_col_max = data.iloc[:, 0].max()
            return data / first_col_max
        else:
            # For Series, divide by maximum value
            return data / data.max()
    
    # Apply normalization if requested
    if normalize_left:
        left_datasets = [normalize_dataset(ds) for ds in left_datasets]
    if normalize_right:
        right_datasets = [normalize_dataset(ds) for ds in right_datasets]
    
    # Slice datasets to the plot window so the y-axis autoscales to visible data only
    # (done after normalization so the normalization baseline stays the full-history max)
    left_datasets = [ds.loc[(ds.index >= plot_start) & (ds.index <= plot_end)] for ds in left_datasets]
    right_datasets = [ds.loc[(ds.index >= plot_start) & (ds.index <= plot_end)] for ds in right_datasets]
    
    # Plot left axis datasets
    left_color = "tab:blue"
    left_colors = plt.cm.Blues(np.linspace(1.0, 0.35, len(left_datasets)))
    ax1.set_xlabel("Date")
    left_labels = []
    for idx, data in enumerate(left_datasets):
        if isinstance(data, pd.DataFrame):
            # Use only the first column's name as the label for the entire DataFrame
            label = str(data.columns[0])
            # Plot first column with label, rest without
            for col_idx, col in enumerate(data.columns):
                if col_idx == 0:
                    ax1.plot(data.index, data[col], color=left_colors[idx], label=label)
                else:
                    ax1.plot(data.index, data[col], color=left_colors[idx], linestyle='--')
            left_labels.append(label)
        else:
            label = getattr(data, "name", None) or f"Left {idx+1}"
            ax1.plot(data.index, data.to_numpy(), color=left_colors[idx], label=label)
            left_labels.append(label)
    
    left_ylabel = "Normalized" if normalize_left else (left_labels[0] if len(left_labels) == 1 else "Left Axis")
    ax1.set_ylabel(left_ylabel, color=left_color)
    ax1.tick_params(axis="y", labelcolor=left_color)
    ax1.grid(True, alpha=0.3)
    title_left = left_ylabel if len(left_labels) == 1 else f"{len(left_labels)} series"
    ax1.set_title(f"{title_left}")

    lines2, labels2 = [], []
    if right_datasets:
        # Plot right axis datasets
        ax2 = ax1.twinx()
        right_color = "tab:orange"
        right_colors = plt.cm.Oranges(np.linspace(1.0, 0.35, len(right_datasets)))
        right_labels = []
        for idx, data in enumerate(right_datasets):
            if isinstance(data, pd.DataFrame):
                # Use only the first column's name as the label for the entire DataFrame
                label = str(data.columns[0])
                # Plot first column with label, rest without
                for col_idx, col in enumerate(data.columns):
                    if col_idx == 0:
                        ax2.plot(data.index, data[col], color=right_colors[idx], alpha=0.7, label=label)
                    else:
                        ax2.plot(data.index, data[col], color=right_colors[idx], alpha=0.7, linestyle='--')
                right_labels.append(label)
            else:
                label = getattr(data, "name", None) or f"Right {idx+1}"
                ax2.plot(data.index, data.to_numpy(), color=right_colors[idx], alpha=0.7, label=label)
                right_labels.append(label)
        
        right_ylabel = "Normalized" if normalize_right else (right_labels[0] if len(right_labels) == 1 else "Right Axis")
        ax2.set_ylabel(right_ylabel, color=right_color)
        ax2.tick_params(axis="y", labelcolor=right_color)
        title_right = right_ylabel if len(right_labels) == 1 else f"{len(right_labels)} series"
        ax1.set_title(f"{title_left} vs {title_right}")
        lines2, labels2 = ax2.get_legend_handles_labels()

    # Draw bear market periods
    if bear_markets is not None:
        for idx, (start, end) in enumerate(bear_markets):
            ax1.axvspan(start, end, alpha=0.1, color="black", label="Bear Markets" if idx == 0 else "")

    lines1, labels1 = ax1.get_legend_handles_labels()

    # Add significant labels to x-axis
    if x_axis_labels is not None:
        # Add vertical lines and labels for each label
        for label_name, label_date in x_axis_labels.items():
            # Filter labels that fall within the plot date range
            if label_date < plot_start or label_date > plot_end:
                continue
            ax1.axvline(x=label_date, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            # Add text label rotated vertically
            ax1.text(label_date, ax1.get_ylim()[1] * 0.98, label_name, 
                    rotation=90, verticalalignment='top', horizontalalignment='right',
                    fontsize=8, alpha=0.8, color='red')
    
    # Set x limits
    ax1.set_xlim(plot_start, plot_end)
    # Combine legends
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Enable interactive mode for non-blocking plots
    plt.ion()

    # Fetch datasets
    cpi_scaling = get_CPI_scaling()
    sp500 = fetch_yfinance('^GSPC').resample("D").ffill().rename("S&P 500 Index")
    buffet = calc_buffett_indicator()
    buffet = fit_exponential(buffet, detrend=True, trends=False)
    cape10 = calc_cape_ratio(averaging_years=10)
    ti10y = fetch_fred_csv("DGS10").rename("10Y Treasury Yield")

    # Detect bear markets in S&P 500
    bear_markets = detect_bear_markets(sp500, threshold=0.2)

    x_axis_labels = FINANCIAL_CRISES

    sp500_1975 = (sp500[sp500.index > dt.datetime(1975, 1, 1)]*cpi_scaling).dropna().rename("S&P 500 Index (CPI-Adjusted)")
    bear_markets_1975 = detect_bear_markets(sp500_1975, threshold=0.2)
    plot_dual_axis(fit_exponential(sp500_1975, detrend=True, trends=True), [], bear_markets_1975, x_axis_labels=x_axis_labels, normalize_right=True)

    # sp500 = fit_exponential(sp500, detrend=True, trends=False)
    plot_dual_axis(sp500, [buffet, cape10, ti10y], bear_markets, x_axis_labels=x_axis_labels, normalize_right=True)

    earnings_ratio = calc_treasury_cape_ratio(averaging_years=10)
    earnings_ratio = add_bands(earnings_ratio)
    plot_dual_axis(sp500, [earnings_ratio], bear_markets, x_axis_labels=x_axis_labels, normalize_right=False)

    plot_dual_axis(sp500, [ti10y], bear_markets, x_axis_labels=x_axis_labels, normalize_right=False)

    excess_cape_yield = calc_excess_cape_yield(averaging_years=10)
    excess_cape_yield5 = calc_excess_cape_yield(averaging_years=5)
    excess_cape_yield4 = calc_excess_cape_yield(averaging_years=4)
    excess_cape_yield3 = calc_excess_cape_yield(averaging_years=3)
    excess_cape_yield2 = calc_excess_cape_yield(averaging_years=2)
    excess_cape_yield1 = calc_excess_cape_yield(averaging_years=1)
    excess_cape_yield = add_bands(excess_cape_yield)
    # plot_dual_axis(sp500, [excess_cape_yield, excess_cape_yield5, excess_cape_yield3, excess_cape_yield1], bear_markets, x_axis_labels=x_axis_labels, normalize_right=False)
    plot_dual_axis(sp500, [excess_cape_yield, excess_cape_yield5, excess_cape_yield4, excess_cape_yield3, excess_cape_yield2, excess_cape_yield1], bear_markets, x_axis_labels=x_axis_labels, normalize_right=False)

    # gold = fetch_yfinance('GC=F').resample("D").ffill().rename("Gold")
    # gold = fit_exponential(gold, detrend=False, trends=True)
    # silver = fetch_yfinance('SI=F').resample("D").ffill().rename("Silver")
    # silver = fit_exponential(silver, detrend=False, trends=True)
    # plot_dual_axis([gold], [], x_axis_labels=FINANCIAL_CRISES, normalize_left=False, normalize_right=False, plot_from="min")

    # alphabet = fetch_yfinance('GOOGL').resample("D").ffill().rename("Alphabet")
    # alphabet = alphabet[alphabet.index >= pd.Timestamp("2015-01-01")]
    # alphabet = fit_exponential(alphabet, detrend=False, trends=True)
    # plot_dual_axis([alphabet], [], x_axis_labels=FINANCIAL_CRISES, normalize_left=False, normalize_right=False, plot_from="min")



    # Keep plots open
    plt.show(block=True)
