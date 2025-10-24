# Market Valuation Tool
Python tool for analysis of stock market valuation. It is designed mainly for the US stock market. It is just a simple python tool with several functions for personal analyses without any guarantee. It allows the user to fetch and analyse up-to-date financial data.

The author is not a registered investment advisers and does not guarantee the accuracy or completeness of the metrics, data, methodology, code or any other part of the repository. Individual investment decisions are best made with the help of a professional investment adviser.

## Supported tickers and metrics for market valuation
- Buffet indicator = Total US Stock Market Value / Gross Domestic Product (GDP)
- (Shiller) CAPE ratio = S&P 500 Cyclically Adjusted Price/Earnings (CAPE) ratio
- 10-year Treasury yield
- 10-year Treasury / CAPE yield ratio  = 10-year Treasury yield / S&P 500 Cyclically Adjusted Earnings yield
- Excess CAPE Yield = S&P 500 Cyclically Adjusted Earnings yield - (10-year Treasury yield - inflation)
- Arbitrary tickers (stocks, indices, ...) from Yahoo Finance

more to come

## Other features
- Automatic bear market/recession detection
- Trend fitting and possible removal
- Error bars
- Dual-axis ploting

## Dependencies
The tool is dependent on some standard and basic libraries + few libraries for fetching up-to-date financial data. Since the APIs to the financial data providers change frequently, it is recommended to use the most up-to-date versions of the `yfinance` and `wbdata` libraries.
