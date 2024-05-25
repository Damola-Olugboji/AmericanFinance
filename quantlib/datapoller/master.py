import os
from dotenv import load_dotenv

load_dotenv()

from quantlib.datapoller.crypto import Crypto
from quantlib.datapoller.equities import Equities
from quantlib.datapoller.macro import Macro


class DataMaster:
    def __init__(
        self,
        config_keys={
            "yfinance": True,
            # "eodhd": os.getenv("EOD_KEY"),
            "binance": (os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET")),
            # "phemex": True,
            # "oanda": ("practice", os.getenv("OANDA_ACC"), os.getenv("OANDA_KEY")),
            # "coinbase": False,
            "polygon": os.get_env("POLYGON_API_KEY"),
            "fred": os.get_env("FRED_API_KEY"),
            "hyperliquid": (os.getenv("HYP_API_WALLET_ADDR"), os.getenv("HYP_API_WALLET_PRIVATE_KEY")),
        },
    ) -> None:
        src_pollers = {}
        if "yfinance" in config_keys and config_keys["yfinance"]:
            from quantlib.wrappers.yfinance import YFinance

            src_pollers["yfinance"] = YFinance()
        if "binance" in config_keys and config_keys["binance"]:
            from quantlib.wrappers.binance import BinanceUS

            src_pollers["binance"] = BinanceUS(config_keys["binance"][0], config_keys["binance"][1])
        if "polygon" in config_keys and config_keys["polygon"]:
            from quantlib.wrappers.polygon import Polygon

            src_pollers["polygon"] = Polygon(config_keys["polygon"])
        if "hyperliquid" in config_keys and config_keys["hyperliquid"]:
            from quantlib.wrappers.hyperliquid import Hyperliquid

            src_pollers["hyperliquid"] = Hyperliquid(config_keys["hyperliquid"][0], config_keys["hyperliquid"][1])
        if "fred" in config_keys and config_keys["fred"]:
            from quantlib.wrappers.fred import Fred

            src_pollers["fred"] = Fred(config_keys["fred"])

        self.crypto = Crypto(pollers=src_pollers, default_src="binance")
        self.equities = Equities(pollers=src_pollers, default_src="polygon")
        self.macro = Macro(pollers=src_pollers, default_src="fred")
