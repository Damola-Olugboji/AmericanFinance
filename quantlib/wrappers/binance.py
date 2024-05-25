from binance.client import Client
from quantlib.utilities.general import standard_to_binance_date

# import time
# import asyncio
# import aiohttp
# import logging
import numpy as np
import pandas as pd

# from io import StringIO
# from copy import deepcopy
# from decimal import Decimal 
# from bs4 import BeautifulSoup
# from collections import defaultdict, deque
# from quantlib.throttler.aiohttp import asession_requests_get

# from binance import streams
# from binance.enums import FuturesType
# from binance import AsyncClient, BinanceSocketManager
# from binance.exceptions import BinanceAPIException

# import quantlib.standards.markets as markets
# from quantlib.standards.intervals import Period
# from quantlib.throttler.rate_semaphore import RateSemaphore, AsyncRateSemaphore

# async def send(self, message):
#     await self.ws.send(message)
#     return
# streams.ReconnectingWebsocket.send = send
# def futures_depth_socket(self, symbol, depth=10, speed_ms = None, futures_type = FuturesType.USD_M):
#     assert speed_ms in [None, 100, 250, 500]
#     speed = f'@{str(speed_ms)}ms' if speed_ms is not None else ''
#     return self._get_futures_socket(symbol.lower() + '@depth' + str(depth) + speed, futures_type = futures_type)
# BinanceSocketManager.futures_depth_socket = futures_depth_socket

# def subscription_to_identifier(subscription) -> str:
#     if subscription["type"] == f"l2_book_{FuturesType.USD_M}":
#         return f'l2_book_{FuturesType.USD_M}:{subscription["symbol"].lower()}'
#     raise Exception()

# async def parse_fr_table():
#     url = "https://www.binance.com/en/futures/funding-history/perpetual/real-time-funding-rate"
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             soup = BeautifulSoup(await response.read(),'lxml')
#             table = soup.find_all('table')[0] 
#             df = pd.read_html(StringIO(str(table)))[0]
#             df.columns = ["Symbol"] + list(df)[1:]
#             df = df.dropna(subset=["Symbol"])
#             df["Symbol"] = df["Symbol"].str.replace(" Perpetual", "")
#             df["Interval"] = df["Interval"].str.replace("h","").astype(int)
#             fr_table = df[["Symbol", "Interval"]].set_index('Symbol').rename(columns={
#                 "Interval": "frint" 
#             }).T.to_dict()
#             assert len(df) > 280
#     logging.info(f"got {len(fr_table)} real-time binance funding rates", extra={"exchange":"bin"})
#     return fr_table

class BinanceUS:
    def __init__(self, api_key, secret_key, **kwargs) -> None:
        self.client = Client(api_key=api_key, api_secret=secret_key, tld="us")
        self.aclient = None
        self.aws_manager = None
        self.conns = {}
        # self.rate_semaphore = AsyncRateSemaphore(2400)
        self.obj_l2_book_subscriptions = set()
        
    # async def init_client(self):
    #     """
    #     Initializes the exchange client
    #     """
    #     if not self.aclient:
    #         self.aclient = await AsyncClient.create(api_key = self.api_key, api_secret = self.api_secret, tld = "us")
    #     if not self.aws_manager:
    #         self.aws_manager = BinanceSocketManager(self.aclient)
            
    def get_ohlcv(self, ticker, start, end, granularity, **kwargs):
        klines = self.client.get_historical_klines(
            symbol=ticker,
            interval=granularity,
            start_str=standard_to_binance_date(start),
            end_str=standard_to_binance_date(end),
        )

        return klines
    
    def is_any_stream_error(self):
        return any(isinstance(task, Exception) for id, task in self.conn.items())

    def get_recent_trades(self, ticker, limit=1000, **kwargs):
        recent_trades = self.client.get_recent_trades(symbol = ticker, limit = limit)
        return pd.DataFrame(recent_trades)

    def get_historical_trades(self, ticker):
        pass
