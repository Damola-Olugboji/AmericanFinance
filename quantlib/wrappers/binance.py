from binance.client import Client
from quantlib.utilities.general import convert_date


class BinanceUS:
    def __init__(self, api_key, secret_key) -> None:
        self.client = Client(api_key=api_key, api_secret=secret_key, tld="us")

    def get_ohlcv(self, ticker, start, end, granularity, **kwargs):
        klines = self.client.get_historical_klines(
            symbol=ticker,
            interval=granularity,
            start_str=convert_date(start),
            end_str=convert_date(end),
        )

        return klines

    def get_recent_trades(self, ticker, limit=1000, **kwargs):
        pass

    def get_historical_trades(self, ticker):
        pass
