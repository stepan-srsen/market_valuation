"""Shared data loading and processing helpers.

Contains:
  - Cache-aware downloaders for yfinance and FRED (refreshed once per day)
  - Loaders for local files: S&P 500 earnings CSV, MSCI index xlsx exports
  - CPI helpers (extrapolation, scaling, inflation) and GDP extension with GDPNow
  - Generic time-series processing: common date range, bear-market detection, std-dev bands,
    exponential trend fitting
"""

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Cache directory for downloaded data
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
# Data directory for historical data
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _cache_is_fresh(cache_path: Path) -> bool:
    """Return True if the cache file exists and was last modified today."""
    if not cache_path.exists():
        return False
    file_mod_time = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
    return file_mod_time.date() == dt.datetime.now().date()


# --- yfinance ---

def fetch_yfinance(ticker="^GSPC", auto_adjust=True, period="max", interval="1d") -> pd.Series:
    """Download a ticker via yfinance (cached locally) and return daily closes."""
    # Create cache filename
    adj_str = "adj" if auto_adjust else "noadj"
    filename = f"{ticker.replace('^', '')}_{interval}_{period}_{adj_str}.pkl"
    cache_path = CACHE_DIR / filename

    # Download if cache is stale
    if _cache_is_fresh(cache_path):
        data = pd.read_pickle(cache_path)
    else:
        data = yf.download(ticker, auto_adjust=auto_adjust, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}.")
        data.to_pickle(cache_path)

    # Handle multi-level columns (ticker level)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"][ticker].dropna()
    else:
        closes = data["Close"].dropna()

    return closes


def fetch_yfinance_monthly(ticker: str, name: str) -> pd.Series:
    """Download daily closes for ticker via yfinance (cached locally) and return monthly closes."""
    closes = fetch_yfinance(ticker, auto_adjust=True, period="max", interval="1d")
    monthly = closes.groupby(closes.index.to_period("M")).last()
    monthly.name = name
    return monthly


# --- FRED ---

def fetch_fred_csv(id: str) -> pd.Series:
    """Fetch a time series in csv from FRED (cached locally, refreshed once per day)."""
    # FRED's python API needs an API key, so I use direct CSV download
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
    filename = id + ".csv"
    cache_path = CACHE_DIR / filename

    # Download if cache is stale
    if _cache_is_fresh(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["observation_date"])
    else:
        df = pd.read_csv(url, parse_dates=["observation_date"])
        df.to_csv(cache_path, index=False)

    df[id] = pd.to_numeric(df[id].replace(".", pd.NA), errors="raise")
    series = df.set_index("observation_date")[id].dropna().sort_index()
    return series.rename(id)

# --- Local / manual data files ---

def fetch_sp500_earnings() -> pd.Series:
    """Fetch S&P 500 earnings data from S&P Global, cache it locally, and combine with historical data."""
    # import httpx  # uncomment together with the download block below (pip install httpx)
    # filename = "sp-500-eps-est.xlsx"
    # cache_path = CACHE_DIR / filename

    # # Check if file exists and was modified today
    # should_download = True
    # if cache_path.exists():
    #     file_mod_time = dt.datetime.fromtimestamp(cache_path.stat().st_mtime)
    #     if file_mod_time.date() == dt.datetime.now().date():
    #         should_download = False

    # # Download if needed (Excel file, not CSV)
    # if should_download:

    #     URL = "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx"
    #     headers = {
    #         # Copy a recent Chrome UA from your machine (DevTools > Network)
    #         "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    #                     "AppleWebKit/537.36 (KHTML, like Gecko) "
    #                     "Chrome/126.0.0.0 Safari/537.36"),
    #         "Accept": "application/octet-stream,*/*",
    #         "Accept-Language": "en-US,en;q=0.9",
    #         "Accept-Encoding": "gzip, deflate, br, zstd",
    #         # Intentionally no Referer because pasting URL in a new tab has none
    #         # Add the “sec-ch-ua*” and “sec-fetch-*” hints many CDNs expect:
    #         "Sec-Fetch-Site": "none",
    #         "Sec-Fetch-Mode": "navigate",
    #         "Sec-Fetch-Dest": "document",
    #         "Pragma": "no-cache",
    #         "Cache-Control": "no-cache",
    #     }
    #     with httpx.Client(http2=True, headers=headers, follow_redirects=True, timeout=60) as client:
    #         r = client.get(URL)
    #         r.raise_for_status()
    #         ct = r.headers.get("Content-Type", "")
    #         # XLSX should be one of these MIME types:
    #         assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in ct or "application/octet-stream" in ct, ct
    #         with open(cache_path, "wb") as f:
    #             f.write(r.content)
    #     print("Saved updated S&P 500 earnings data.")

    # df = pd.read_excel(cache_path, sheet_name='QUARTERLY DATA', header=None, skiprows=6, usecols=[0,2], index_col=0, parse_dates=True).squeeze()
    # df = df.dropna().sort_index() # drop missing values and sort
    # df = pd.to_numeric(df, errors='raise') # ensure earnings are numeric
    # df = df.rolling(window=4).sum().dropna() # calculate trailing 12-month (4 quarters)

    # Load historical data from local CSV
    df_history = pd.read_csv(DATA_DIR / "SP500EARNINGS.csv", index_col=0, parse_dates=True, dayfirst=True).squeeze()
    df_history = df_history.dropna().sort_index() # drop missing values and sort
    df_history = pd.to_numeric(df_history, errors='raise') # ensure earnings are numeric
    # df_history = df_history[df_history.index < df.index[0]] # take only historical data before fetched data

    # # Combine historical data with fetched data
    # df = pd.concat([df_history, df])

    df = df_history  # Use only historical data for now, as fetching from S&P was discontinued

    # Set series name
    df.name = "SP500_Earnings"
    return df


def load_msci_index(path: Path) -> pd.Series:
    """Load a monthly MSCI index export (xlsx) and return a Series indexed by month Period."""
    df = pd.read_excel(path, header=5, usecols=[0, 1])
    df.columns = ["Date", "Price"]
    df = df.dropna(subset=["Date", "Price"])
    df["Date"] = pd.to_datetime(df["Date"])
    name = Path(path).name.split(" - ")[1]  # e.g. "MSCI World Value Index"
    series = pd.Series(df["Price"].values, index=df["Date"].dt.to_period("M"), name=name)
    return series


# --- CPI and inflation helpers ---

def get_cpi(extrapolate=True, ema_span=12, interpolate=True) -> pd.Series:
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
        # extrapolated_values = [cpi.iloc[-1] * (1 + ema_growth) ** (1-2**(-i-1)) for i in range(extrapolate)] # ultraconservative version
        # Create series and append
        extrapolated_series = pd.Series(extrapolated_values, index=extrapolated_dates)
        cpi = pd.concat([cpi, extrapolated_series])
    # Interpolate to get smooth evolution
    if interpolate:
        cpi = cpi.resample('D').interpolate(method='linear')
    # Limit to current date
    cpi = cpi[cpi.index <= pd.Timestamp.now()]
    return cpi.rename("CPI")


def get_CPI_scaling() -> pd.Series:
    """Get CPI scaling factor (latest CPI / historical CPI) for inflation adjustment."""
    cpi = get_cpi()
    cpi_scaling = cpi.iloc[-1] / cpi
    return cpi_scaling.rename("CPI Scaling")


def get_inflation(averaging_years=10) -> pd.Series:
    """Get the annual inflation rate based on CPI averaged over the last `span` years."""
    cpi = get_cpi()
    ratio = cpi / cpi.shift(int(averaging_years*365))
    ratio_annualized = ratio.dropna() ** (1/averaging_years)
    inflation = ratio_annualized - 1.0
    return inflation.rename("Inflation")


# --- GDP ---

def get_gdp() -> pd.Series:
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


# --- Generic time-series processing ---

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


def add_bands(series: pd.Series, std_devs: list = [1, 2]) -> pd.DataFrame:
    """Add standard deviation bands to a time series.
    
    Args:
        series: Time series to add bands to
        std_devs: List of standard deviation multiples to add bands for
        
    Returns:
        DataFrame with original series and std dev bands
    """
    mean = series.mean()
    std_dev = np.std(series)
    bands = [series]
    bands.append(pd.Series(mean, index=series.index, name=f"{series.name} fit"))
    for n in std_devs:
        bands.append(pd.Series(mean + n * std_dev, index=series.index, name=f"{series.name} +{n} SD"))
        bands.append(pd.Series(mean - n * std_dev, index=series.index, name=f"{series.name} -{n} SD"))
    result = pd.concat(bands, axis=1)
    
    return result


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
    detrended_y = series / y_fit
       
    if detrend:
        detrended_y.name = f"{series.name} (exp detrended)"
        if not trends:
            return detrended_y
        return add_bands(detrended_y)
    else:
        if not trends:
            return series
        return add_bands(detrended_y).mul(y_fit, axis=0)
