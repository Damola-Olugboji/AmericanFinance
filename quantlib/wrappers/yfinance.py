import yfinance as yf
import pandas as pd

class YFinance:
    def __init__(self) -> None:
        pass
    
    def get_ohlcv(self, ticker, start, end, granularity, multiplier, **kwargs):
        df = yf.Ticker(ticker).history(start = start, end = end, interval = granularity, auto_adjust = True).reset_index()
        df.Date = df.Date.dt.tz_convert("UTC")
        df = df.rename(columns={"Date": "datetime", "Open" : "open", "High": "high", "Low" : "low", "Close" : "close", "Volume": "volume"})
        if df.empty:
            return pd.DataFrame()
        df = df.drop(columns=["Dividends", "Stock Splits"])
        df = df.set_index("datetime",drop=True)
        return df.loc[start:end]   
    