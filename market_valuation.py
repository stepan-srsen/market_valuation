import math
import datetime as dt
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import wbdata
import yfinance as yf
import numpy as np
import httpx

# TODO: plot excess cape yield
# TODO: CPI adjusted treasury yield? Check Shiller's work
# TODO: add x-axis labels with financial crises (e.g. dot-com bubble in 2000)
# TODO: ?implement exponential moving average for smoothing inflation
# TODO: ?add seasonal trends to CPI prediction from historical seasonally adjusted vs non-adjusted data

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
# Data directory for historical data
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def fetch_yfinance(ticker="^GSPC", auto_adjust=True, period="max", interval="1d") -> pd.Series:
    """Download the S&P 500 index and return daily closes."""
    # Create cache filename
    adj_str = "adj" if auto_adjust else "noadj"
    filename = f"{ticker.replace('^', '')}_{interval}_{period}_{adj_str}.pkl"
    cache_path = CACHE_DIR / filename
    
    # Check if file exists and was modified today
    should_download = True
    if cache_path.exists():
        file_mod_time = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_mod_time.date() == dt.datetime.now().date():
            should_download = False
    
    # Download if needed
    if should_download:
        data = yf.download(ticker, auto_adjust=auto_adjust, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}.")
        data.to_pickle(cache_path)
    else:
        data = pd.read_pickle(cache_path)
    
    # Handle multi-level columns (ticker level)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"][ticker].dropna()
    else:
        closes = data["Close"].dropna()
    
    return closes

def fetch_wbdata() -> pd.Series:
    """Fetch market cap and GDP data from World Bank."""
    # only annual data available so not used anymore
    COUNTRY_ISO = "USA"
    GDP_IND = "NY.GDP.MKTP.CD"
    MARKET_CAP_IND = "CM.MKT.LCAP.CD"
    indicators = {MARKET_CAP_IND: "market_cap", GDP_IND: "gdp"}
    raw = wbdata.get_dataframe(indicators, country=COUNTRY_ISO)
    raw = raw.sort_index()
    return raw

def fetch_fred_csv(id: str) -> pd.Series:
    """Fetch a time series in csv from FRED and cache it locally."""
    # FRED's python API needs an API key, so I use direct CSV download
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
    filename = id + ".csv"
    cache_path = CACHE_DIR / filename
    
    # Check if file exists and was modified today
    should_download = True
    if cache_path.exists():
        file_mod_time = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_mod_time.date() == dt.datetime.now().date():
            should_download = False
    
    # Download if needed
    if should_download:
        df = pd.read_csv(url, parse_dates=["observation_date"])
        df.to_csv(cache_path, index=False)
    else:
        df = pd.read_csv(cache_path, parse_dates=["observation_date"])
    
    df[id] = pd.to_numeric(df[id].replace(".", pd.NA), errors="raise")
    series = df.set_index("observation_date")[id].dropna().sort_index()
    return series.rename(id)

def fetch_sp500_earnings() -> pd.Series:
    """Fetch S&P 500 earnings data from S&P Global, cache it locally, and combine with historical data."""
    filename = "sp-500-eps-est.xlsx"
    cache_path = CACHE_DIR / filename
    
    # Check if file exists and was modified today
    should_download = True
    if cache_path.exists():
        file_mod_time = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_mod_time.date() == dt.datetime.now().date():
            should_download = False
    
    # Download if needed (Excel file, not CSV)
    if should_download:

        URL = "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx"
        headers = {
            # Copy a recent Chrome UA from your machine (DevTools > Network)
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/126.0.0.0 Safari/537.36"),
            "Accept": "application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            # Intentionally no Referer because pasting URL in a new tab has none
            # Add the “sec-ch-ua*” and “sec-fetch-*” hints many CDNs expect:
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
        with httpx.Client(http2=True, headers=headers, follow_redirects=True, timeout=60) as client:
            r = client.get(URL)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            # XLSX should be one of these MIME types:
            assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in ct or "application/octet-stream" in ct, ct
            with open(cache_path, "wb") as f:
                f.write(r.content)
        print("Saved updated S&P 500 earnings data.")

    df = pd.read_excel(cache_path, sheet_name='QUARTERLY DATA', header=None, skiprows=6, usecols=[0,2], index_col=0, parse_dates=True).squeeze()
    df = df.dropna().sort_index() # drop missing values and sort
    df = pd.to_numeric(df, errors='raise') # ensure earnings are numeric
    df = df.rolling(window=4).sum().dropna() # calculate trailing 12-month (4 quarters)

    # Load historical data from local CSV
    df_history = pd.read_csv(DATA_DIR / "SP500EARNINGS.csv", index_col=0, parse_dates=True, dayfirst=True).squeeze()
    df_history = df_history.dropna().sort_index() # drop missing values and sort
    df_history = pd.to_numeric(df_history, errors='raise') # ensure earnings are numeric
    df_history = df_history[df_history.index < df.index[0]] # take only historical data before fetched data

    # Combine historical data with fetched data
    df = pd.concat([df_history, df])

    # Set series name
    df.name = "SP500_Earnings"
    return df

def fetch_CPI(extrapolate=True, ema_span=12, interpolate=True) -> pd.Series:
    """Fetch Consumer Price Index (CPI) data from FRED and process it."""
    # Fetch CPI data
    cpi = fetch_fred_csv("CPIAUCNS")
    # Shift to middle of month
    cpi = cpi.resample('MS').mean()
    cpi.index = cpi.index + pd.Timedelta(days=14)
    if extrapolate is True:
        # Calculate number of months to extrapolate from last CPI date to today
        days_diff = (pd.Timestamp.now() - cpi.index[-1]).days
        extrapolate = int(math.ceil(days_diff / 28)) # rather more than less
    if extrapolate > 0:
        # Extrapolate n_extrapolate additional months using exponential moving average on the trend
        growth_rates = cpi.pct_change().dropna() # month-over-month growth rates
        ema_growth = growth_rates.ewm(span=ema_span).mean().iloc[-1] # exponential moving average of growth rate
        # Generate extrapolated values
        extrapolated_dates = pd.date_range(start=cpi.index[-1] + pd.DateOffset(months=1), periods=extrapolate, freq=pd.DateOffset(months=1))
        # extrapolated_values = [cpi.iloc[-1] * (1 + ema_growth) ** (i + 1) for i in range(extrapolate)]
        extrapolated_values = [cpi.iloc[-1] * (1 + ema_growth) ** (2-2**(-i)) for i in range(extrapolate)] # conservative version
        # extrapolated_values = [cpi.iloc[-1] * (1 + ema_growth) ** (1-2**(-i-1)) for i in range(extrapolate)] # 2nd conservative version
        # Create series and append
        extrapolated_series = pd.Series(extrapolated_values, index=extrapolated_dates)
        cpi = pd.concat([cpi, extrapolated_series])
    # Interpolate to get smooth evolution
    if interpolate:
        cpi = cpi.resample('D').interpolate(method='linear')
    # Limit to current date
    cpi = cpi[cpi.index <= pd.Timestamp.now()]
    return cpi.rename("CPI")

def fetch_inflation(averaging_years=10) -> float:
    """Fetch the inflation rate based on CPI averaged over the last `span` months."""
    cpi = fetch_CPI()
    ratio = cpi / cpi.shift(averaging_years*365)
    ratio_anualized = ratio.dropna() ** (1/averaging_years)
    inflation = ratio_anualized - 1.0
    return inflation.rename("Inflation")

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
    cpi = fetch_CPI()
    
    # Align all series to the same index
    earnings = earnings.reindex(prices.index, method='ffill')
    cpi = cpi.reindex(prices.index, method='ffill')

    # Adjust for inflation (normalize to latest CPI value)
    latest_cpi = cpi.iloc[-1]
    real_prices = prices * (latest_cpi / cpi)
    real_earnings = earnings * (latest_cpi / cpi)

    # Calculate rolling average of real earnings
    avg_real_earnings = real_earnings.rolling(window=averaging_years * 365, min_periods=averaging_years * 365).mean()
    
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
    inflation = fetch_inflation(averaging_years=averaging_years)
    # Align all series to the same index
    ti10y = ti10y.reindex(cape.index, method='ffill')
    inflation = inflation.reindex(cape.index, method='ffill')
    # Calculate excess CAPE yield and drop NA values
    excess_cape_yield = (1.0/cape - ti10y/100.0 + inflation).dropna()
    return excess_cape_yield.rename(f"Excess CAPE Yield ({averaging_years}yr)")

def fetch_us_gdp_and_gdpnow() -> pd.Series:
    """Fetch US GDP and extend it with GDPNow estimates."""
    # alternatively, there is python API but it requires an API key
    # fetch GDP and GDPNOW from FRED
    gdp = fetch_fred_csv("GDP")
    gdp_now_annualized = fetch_fred_csv("GDPNOW")
    # take only GDPNOW entries after last GDP date
    last_gdp_date = gdp.index.max()
    gdpnow_new = gdp_now_annualized[gdp_now_annualized.index > last_gdp_date]
    if gdpnow_new.empty:
        return gdp
    last_gdp = gdp.iloc[-1]
    # Calculate actual time periods in years from previous point
    days_elapsed = gdpnow_new.index.to_series().diff().fillna(gdpnow_new.index[0] - last_gdp_date)
    years_elapsed = days_elapsed.dt.days / 365
    # Convert annualized growth rates to actual period growth factors
    growth_factors = (1 + gdpnow_new.div(100.0)) ** years_elapsed.values
    gdp_extension = (last_gdp * growth_factors.cumprod()).rename("GDP")
    extended_gdp = pd.concat([gdp, gdp_extension])
    return extended_gdp

def calc_buffett_indicator() -> pd.Series:
    """Calculate the Buffett Indicator (Market Cap / GDP ratio).
    
    Args:
        exponential_fit: If True, fit an exponential trend and return the ratio
                        divided by the exponential fit (detrended ratio).
        detrend: If True and exponential_fit is True, return detrended data
    """
    # Fetch market cap data
    market_cap = fetch_yfinance('^FTW5000').resample('D').ffill()
    # Fetch GDP data with GDPNow extension
    gdp_data = fetch_us_gdp_and_gdpnow()
    # Align GDP to market cap dates
    gdp_aligned = gdp_data.reindex(market_cap.index, method='ffill')
    # Calculate ratio
    ratio = market_cap / gdp_aligned
    ratio.name = "Buffett Indicator"
    
    return ratio.dropna()

def detect_bear_markets(series: pd.Series, threshold: float = 0.20) -> list:
    """Detect bear market periods (decline of threshold% or more from peak)."""
    cleaned_series = series.dropna()
    if cleaned_series.empty:
        return []

    bear_periods = []
    max_price = cleaned_series.iloc[0]
    max_date = cleaned_series.index[0]
    min_price, min_date = None, None
    in_bear = False
    start = None

    for date, price in cleaned_series.iloc[1:].items():
        if price >= max_price:
            if in_bear:
                bear_periods.append((start, min_date))
                in_bear = False
            max_price = price
            max_date = date
            continue
        if in_bear and price < min_price:
            min_price = price
            min_date = date
            continue
        drawdown = (max_price - price) / max_price
        if drawdown >= threshold and not in_bear:
            in_bear = True
            start = max_date
            min_price = price
            min_date = date
    if in_bear:
        bear_periods.append((start, cleaned_series.index[-1]))

    return bear_periods

def fit_exponential(series: pd.Series, detrend: bool = False, trends: bool = True) -> pd.DataFrame:
    """Fit an exponential trend to a time series and return with confidence bands.
    
    Args:
        series: Time series to fit
        detrend: If True, return detrended data, otherwise return original with fit overlay
        
    Returns:
        DataFrame with original/detrended series, exponential fit, and std dev bands
    """
    # Convert dates to numeric (days since start)
    x = (series.index - series.index[0]).days.values
    y = series.values
    
    # Fit exponential: y = a * exp(b * x)
    # Using log transform: log(y) = log(a) + b * x
    log_y = np.log(y)
    coeffs = np.polyfit(x, log_y, 1)
    b, log_a = coeffs[0], coeffs[1]
    
    # Calculate fitted values
    y_fit = np.exp(log_a + b * x)
    
    # Detrend the data first
    detrended_y = y / y_fit
    
    # Calculate residuals and standard deviation from detrended data
    residuals = detrended_y - 1.0  # Mean of detrended data is 1.0
    std_dev = np.std(residuals)
    
    if detrend:
        series = pd.Series(detrended_y, index=series.index, name=f"{series.name} (detrended)")
        if not trends:
            return series
        base_trend = pd.Series(np.ones_like(y_fit), index=series.index, name=f"{series.name} exp fit")
    else:
        if not trends:
            return series
        base_trend = pd.Series(y_fit, index=series.index, name=f"{series.name} exp fit")
    bands = [series, base_trend]
    std_devs = [1, 2]
    for n in std_devs:
        bands.append(pd.Series(base_trend * (1 + n * std_dev), index=series.index, name=f"{series.name} exp +{n} SD"))
        bands.append(pd.Series(base_trend * (1 - n * std_dev), index=series.index, name=f"{series.name} exp -{n} SD"))
    result = pd.concat(bands, axis=1)
    
    return result

def common_date_range(*datasets):
    """Return copies of datasets limited to their shared date range."""
    filter_max = True # filter to the earliest common end date?
    if not datasets:
        return []
    min_dates, max_dates = [], []
    for data in datasets:
        if data.empty:
            raise ValueError("Datasets must be non-empty.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("All datasets must use a DatetimeIndex.")
        min_dates.append(data.index.min())
        max_dates.append(data.index.max())
    common_start = max(min_dates)
    if filter_max:
        common_end = min(max_dates)
    else:
        common_end = max(max_dates)
    if common_start > common_end:
        raise ValueError("Datasets do not share a common date range.")
    return [data.loc[(data.index >= common_start) & (data.index <= common_end)] for data in datasets]

def plot_dual_axis(left_dataset, right_dataset, bear_markets=None, normalize_left=False, normalize_right=False) -> None:
    """Plot datasets on dual y-axes with optional normalization.
    
    Args:
        left_dataset: Single Series/DataFrame or list of Series/DataFrames for left axis
        right_dataset: Single Series/DataFrame or list of Series/DataFrames for right axis
        bear_markets: List of (start, end) tuples for bear market periods
        normalize_left: If True, normalize left axis datasets to their maximum
        normalize_right: If True, normalize right axis datasets to their maximum
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Draw bear market periods
    if bear_markets:
        for idx, (start, end) in enumerate(bear_markets):
            ax1.axvspan(start, end, alpha=0.2, color="gray", label="Bear Markets" if idx == 0 else "")
    
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
    
    # Convert single datasets to lists
    left_datasets = left_dataset if isinstance(left_dataset, list) else [left_dataset]
    right_datasets = right_dataset if isinstance(right_dataset, list) else [right_dataset]
    
    # Apply normalization if requested
    if normalize_left:
        left_datasets = [normalize_dataset(ds) for ds in left_datasets]
    if normalize_right:
        right_datasets = [normalize_dataset(ds) for ds in right_datasets]
    
    # Plot left axis datasets
    left_color = "tab:blue"
    left_colors = plt.cm.Blues(np.linspace(1.0, 0.4, len(left_datasets)))
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
                    ax1.plot(data.index, data[col], color=left_colors[idx])
            left_labels.append(label)
        else:
            label = getattr(data, "name", None) or f"Left {idx+1}"
            data.plot(ax=ax1, color=left_colors[idx], label=label, legend=False)
            left_labels.append(label)
    
    left_ylabel = "Normalized" if normalize_left else (left_labels[0] if len(left_labels) == 1 else "Left Axis")
    ax1.set_ylabel(left_ylabel, color=left_color)
    ax1.tick_params(axis="y", labelcolor=left_color)
    ax1.grid(True, alpha=0.3)

    # Plot right axis datasets
    ax2 = ax1.twinx()
    right_color = "tab:red"
    right_colors = plt.cm.Reds(np.linspace(1.0, 0.4, len(right_datasets)))
    
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
                    ax2.plot(data.index, data[col], color=right_colors[idx], alpha=0.7)
            right_labels.append(label)
        else:
            label = getattr(data, "name", None) or f"Right {idx+1}"
            data.plot(ax=ax2, color=right_colors[idx], alpha=0.7, label=label, legend=False)
            right_labels.append(label)
    
    right_ylabel = "Normalized" if normalize_right else (right_labels[0] if len(right_labels) == 1 else "Right Axis")
    ax2.set_ylabel(right_ylabel, color=right_color)
    ax2.tick_params(axis="y", labelcolor=right_color)
    
    # Set title
    title_left = left_ylabel if len(left_labels) == 1 else f"{len(left_labels)} series"
    title_right = right_ylabel if len(right_labels) == 1 else f"{len(right_labels)} series"
    ax1.set_title(f"{title_left} vs {title_right}")
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Fetch both datasets
    sp500 = fetch_yfinance('^GSPC', auto_adjust=True).rename("S&P 500 Index")
    buffet = calc_buffett_indicator()
    buffet = fit_exponential(buffet, detrend=True, trends=False)
    cape10 = calc_cape_ratio(averaging_years=10).resample('D').ffill()
    ti10y = fetch_fred_csv("DGS10").resample('D').ffill().rename("10Y Treasury Yield")
    print(sp500)
    print(buffet)
    print(cape10)
    print(ti10y)

    # Detect bear markets in S&P 500
    bear_markets = detect_bear_markets(sp500, threshold=0.2)

    plot_dual_axis(sp500, [buffet, cape10, ti10y], bear_markets, normalize_right=True)

    earnings_ratio = calc_treasury_cape_ratio(averaging_years=10)
    plot_dual_axis(sp500, [earnings_ratio], bear_markets, normalize_right=False)

    excess_cape_yield = calc_excess_cape_yield(averaging_years=10)
    plot_dual_axis(sp500, [excess_cape_yield], bear_markets, normalize_right=False)