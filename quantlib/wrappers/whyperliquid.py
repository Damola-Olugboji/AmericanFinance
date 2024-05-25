import pandas as pd
import numpy as np
import requests
import json
import pytz
import asyncio
import logging

from decimal import Decimal
from datetime import date, datetime, timedelta
from collections import deque, defaultdict

import aiohttp
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.api import API
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from hyperliquid.utils.error import ClientError, ServerError
from hyperliquid.utils.signing import (
    CancelRequest,
    OrderRequest,
    OrderType,
    OrderWire,
    get_timestamp_ms,
    order_request_to_order_wire,
    order_wires_to_order_action,
    sign_l1_action,
)
from hyperliquid.utils.types import Any, List, Optional, Cloid
from hyperliquid.websocket_manager import WebsocketManager

from json import JSONDecodeError


def post(self, url_path: str, payload: Any = None) -> Any:  # type: ignore
    payload = payload or {}
    url = self.base_url + url_path
    try:
        response = self.session.post(url, json=payload)
    except requests.ConnectionError:
        self.session.close()
        self.session = requests.Session()
        response = self.session.post(url, json=payload)
    self._handle_exception(response)
    try:
        return response.json()
    except ValueError:
        return {"error": f"Could not parse JSON: {response.text}"}


API.post = post


async def apost(self, url_path, payload):
    async with aiohttp.ClientSession() as asession:
        payload = payload or {}
        url = self.base_url + url_path
        response = await asession.post(url, json=payload)
        await self._ahandle_exception(response)
        try:
            return await response.json()
        except ValueError:
            return {"error": f"Could not parse JSON: {response.text}"}


API.apost = apost


async def _ahandle_exception(self, response):
    status_code = response.status
    if status_code < 400:
        return
    if 400 <= status_code < 500:
        try:
            err_text = await response.text()
            err = json.loads(err_text)
        except JSONDecodeError:
            raise ClientError(status_code, None, err_text, None, response.headers)
        error_data = None
        if "data" in err:
            error_data = err["data"]
        raise ClientError(
            status_code, err["code"], err["msg"], response.headers, error_data
        )
    raise ServerError(status_code, err_text)


API._ahandle_exception = _ahandle_exception


async def clean(self):
    if getattr(self, "assession", None):
        await self.asession.__axit__()


API.clean = clean

"""
exchange.py
"""


async def _apost_action(self, action, signature, nonce):
    payload = {
        "action": action,
        "nonce": nonce,
        "signature": signature,
        "vaultAddress": self.vault_address,
    }
    logging.debug(payload)
    return await self.apost("/exchange", payload)


Exchange._apost_action = _apost_action


async def aorder(
    self,
    coin: str,
    is_buy: bool,
    sz: float,
    limit_px: float,
    order_type: OrderType,
    reduce_only: bool = False,
    cloid: Optional[Cloid] = None,
):
    order: OrderRequest = {
        "coin": coin,
        "is_buy": is_buy,
        "sz": sz,
        "limit_px": limit_px,
        "order_type": order_type,
        "reduce_only": reduce_only,
    }
    if cloid:
        order["cloid"] = cloid
    return await self.abulk_orders([order])


Exchange.aorder = aorder


async def abulk_orders(self, order_requests: List[OrderRequest]) -> Any:  # type: ignore
    order_wires: List[OrderWire] = [
        order_request_to_order_wire(order, self.coin_to_asset[order["coin"]])
        for order in order_requests
    ]
    timestamp = get_timestamp_ms()

    order_action = order_wires_to_order_action(order_wires)

    signature = sign_l1_action(
        self.wallet,
        order_action,
        self.vault_address,
        timestamp,
        self.base_url == constants.MAINNET_API_URL,
    )

    return await self._apost_action(
        order_action,
        signature,
        timestamp,
    )


Exchange.abulk_orders = abulk_orders


async def acancel(self, coin: str, oid: int) -> Any:  # type: ignore
    return await self.abulk_cancel([{"coin": coin, "oid": oid}])


Exchange.acancel = acancel


async def abulk_cancel(self, cancel_requests: List[CancelRequest]) -> Any:  # type: ignore
    timestamp = get_timestamp_ms()
    cancel_action = {
        "type": "cancel",
        "cancels": [
            {
                "a": self.coin_to_asset[cancel["coin"]],
                "o": cancel["oid"],
            }
            for cancel in cancel_requests
        ],
    }
    signature = sign_l1_action(
        self.wallet,
        cancel_action,
        self.vault_address,
        timestamp,
        self.base_url == constants.MAINNET_API_URL,
    )

    return await self._apost_action(
        cancel_action,
        signature,
        timestamp,
    )


Exchange.abulk_cancel = abulk_cancel

from quantlib.throttler.aiohttp import asession_requests_post
from quantlib.throttler.rate_semaphore import RateSemaphore, AsyncRateSemaphore
from quantlib.standards.standards import Period


def get_step_size(szDecimals):
    if szDecimals == 0:
        return Decimal("1")
    else:
        return Decimal("0.1") ** szDecimals


def map_to_hyp_granularities(granularity, granularity_multiplier):
    if granularity == Period.SECOND:
        raise Exception("unsupported granularity in eodhd")
    if granularity == Period.MINUTE:
        assert granularity_multiplier in [1, 3, 5, 15, 30]
        return str(granularity_multiplier) + "m", granularity_multiplier * 60
    if granularity == Period.HOURLY:
        assert granularity_multiplier in [1, 2, 4, 8, 12]
        return str(granularity_multiplier) + "h", granularity_multiplier * 60 * 60
    if granularity == Period.DAILY:
        assert granularity_multiplier in [1, 3]
        return str(granularity_multiplier) + "d", granularity_multiplier * 24 * 60 * 60
    if granularity == Period.WEEKLY:
        assert granularity_multiplier == 1
        return (
            str(granularity_multiplier) + "w",
            granularity_multiplier * 7 * 24 * 60 * 60,
        )
    if granularity == Period.MONTHLY:
        assert granularity_multiplier == 1
        return (
            str(granularity_multiplier) + "m",
            granularity_multiplier * 30 * 7 * 24 * 60 * 60,
        )
    if granularity == Period.YEARLY:
        raise Exception("unsupported granularity in hyperliquid")


class Hyperliquid:
    def __init__(self, address, hyp_key=None, buffer_len=100):
        """
        Initializes the Hyperliquid instance.

        Args:
            address (str): The on-chain address in 42-character hexadecimal format.
            hyp_key (str, optional): The Hyperliquid API key. Defaults to None.
            buffer_len (int, optional): The maximum length of the stream buffer. Defaults to 100.
        """
        self.address = address
        self.info = Info(base_url=constants.MAINNET_API_URL, skip_ws=True)
        self.ws_manager = None
        self.stream_buffer = defaultdict(lambda: deque(maxlen=buffer_len))
        self.aws_manager = AsyncWebsocketManager(
            base_url=constants.MAINNET_API_URL, stream_buffer=self.stream_buffer
        )

        self.wallet = Account.from_key(hyp_key) if hyp_key else None
        self.exchange = Exchange(
            wallet=self.wallet,
            base_url=constants.MAINNET_API_URL,
            account_address=self.wallet.address,
        )
        self.book_manager = BookManager(address=address)

        self.rate_semaphore = RateSemaphore(1200)
        self.arate_semaphore = AsyncRateSemaphore(1200)
        self.hook = None
        self.book_tracked = False

    def init_threaded_ws_manager(self, start=True):
        """
        Initializes the Threaded WebSocket manager thread in daemon mode.

        Args:
            start (bool, optional): Whether to start the WebSocket manager thread. Defaults to True.

        Returns:
            (WebsocketManager): The WebSocket manager instance.
        """
        self.ws_manager = WebsocketManager(base_url=constants.MAINNET_API_URL)
        self.ws_manager.daemon = True
        if start:
            self.ws_manager.start()
        return self.ws_manager

    async def init_client(self, keep_book=True):
        if keep_book:
            await self.keep_book_state()
        return

    async def get_perps_data(self, in_decimal=True):
        """
        Retrieves perpetuals data asynchronously.

        Returns:
            (dict): A dictionary containing perp data with the following structure:

                - "baseAsset" `str` : The base asset symbol.
                - "fr" `float` : Funding rate.
                - "frint" `float`,                Floating point representation of the funding rate.
                - "marginAsset" `str`,            The margin asset symbol.
                - "markPrice" `float`,            The mark price.
                - "minQty" `Decimal`,            The minimum quantity.
                - "min_notional" `float`,         The minimum notional value.
                - "next_funding" `int`,           Unix timestamp of the next funding, milliseconds.
                - "pricePrecision" `Decimal`,     The price precision.
                - "quantityPrecision" `Decimal`,  The quantity precision.
                - "quoteAsset" `str`,             The quote asset symbol.
                - "stepSize" `Decimal`,           The step size.
                - "symbol" `str`,                 The symbol.
                - "timestamp" `int`,              Unix timestamp of the data retrieval, milliseconds.
                - "exchange" `str`                The exchange code , "hyp"

        """
        perps_data = (
            await asession_requests_post(
                urls=["https://api.hyperliquid.xyz/info"],
                payloads=[{"type": "metaAndAssetCtxs"}],
                asemaphore=self.arate_semaphore,
                costs=[2],
                refunds_in=[60],
            )
        )[0]
        logging.info(
            f"got {len(perps_data[1])} @ info/metaAndAssetCtxs endpoint",
            extra={"exchange": "hyp"},
        )
        universe_meta, universe_ctx = perps_data[0]["universe"], perps_data[1]
        symbols_data = {}
        for meta, context in zip(universe_meta, universe_ctx):
            baseAsset = meta["name"]
            fr = np.float64(context["funding"])
            frint = np.float64(1)
            marginAsset = "USDC"
            markPrice = np.float64(context["markPx"])
            minQty = (
                Decimal(get_step_size(meta["szDecimals"]))
                if in_decimal
                else str(get_step_size(meta["szDecimals"]))
            )
            min_notional = np.float64("10")
            _now = datetime.now(pytz.utc)
            _next_hr = (datetime.now(pytz.utc) + timedelta(hours=1)).replace(
                microsecond=0, second=0, minute=0
            )
            next_funding = int(_next_hr.timestamp()) * 1000
            pricePrecision = Decimal("6") if in_decimal else "6"
            quantityPrecision = (
                Decimal(meta["szDecimals"]) if in_decimal else str(meta["szDecimals"])
            )
            quoteAsset = "USDT"
            stepSize = minQty
            maxLeverage = meta["maxLeverage"]
            symbol = meta["name"]
            timestamp = int(_now.timestamp()) * 1000

            symbols_data[symbol] = {
                "baseAsset": baseAsset,
                "fr": fr,
                "frint": frint,
                "marginAsset": marginAsset,
                "markPrice": markPrice,
                "minQty": minQty,
                "min_notional": min_notional,
                "next_funding": next_funding,
                "pricePrecision": pricePrecision,
                "quantityPrecision": quantityPrecision,
                "quoteAsset": quoteAsset,
                "stepSize": stepSize,
                "maxLeverage": maxLeverage,
                "symbol": symbol,
                "timestamp": timestamp,
                "exchange": "hyp",
            }
        logging.info(f"get_perps_data", extra={"exchange": "hyp"})
        return symbols_data

    def get_trade_bars(
        self,
        ticker,
        start,
        end,
        granularity,
        granularity_multiplier,
        kline_close=False,
        **kwargs,
    ):
        """
        Retrieve trade bars data.

        Args:
            ticker (str): Ticker symbol for the asset.
            start (datetime): Start datetime for the data retrieval.
            end (datetime): End datetime for the data retrieval.
            granularity (quantpylib.standards.Period): Granularity of the data.
            granularity_multiplier (int): Multiplier for the granularity.
            exchange (str, optional): Exchange name. Defaults to "US".
            **kwargs: Additional keyword arguments.

        Returns:
            (pd.DataFrame): DataFrame containing the trade bars data.
        """
        gran_str, gran_secs = map_to_hyp_granularities(
            granularity=granularity, granularity_multiplier=granularity_multiplier
        )
        allowed_sec_span = 4990 * gran_secs
        if start.tzinfo is None:
            start = start.replace(tzinfo=pytz.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=pytz.utc)
        requested_span = end.timestamp() - start.timestamp()
        if requested_span > allowed_sec_span:
            max_allowable = datetime.fromtimestamp(end.timestamp() - allowed_sec_span)
            raise Exception(
                f"maximum of 4990 candle periods are allowed, reduce request window to start : {str(max_allowable)}"
            )
        res = self.rate_semaphore.transact(
            lambda_func=lambda: self.candles_snapshot(
                coin=ticker,
                interval=gran_str,
                startms=int(start.timestamp() * 1000),
                endms=int(end.timestamp() * 1000),
            ),
            credits=2,
            refund_time=60,
        )
        df = pd.DataFrame(res)
        if df.empty:
            return pd.DataFrame()
        dt_col = {"T": "datetime"} if kline_close else {"t": "datetime"}
        df = df.rename(
            columns={
                **dt_col,
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        df = df.drop(
            columns=[
                col
                for col in list(df)
                if col not in ["open", "high", "low", "close", "volume", "datetime"]
            ]
        )
        df.datetime = pd.to_datetime(df.datetime, unit="ms", utc=True)
        df = df.set_index("datetime", drop=True)
        df = df[start:end]
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(np.float64)
        return df

    def user_state(self):
        """
        Retrieve trading details about a user.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals#retrieve-users-perpetuals-account-summary
        """
        return self.info.user_state(address=self.address)

    def spot_user_state(self):
        """
        Retrieve trading details about a user in the spot market.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/spot#retrieve-a-users-token-balances
        """
        return self.info.spot_user_state(address=self.address)

    def _open_orders(self):
        """
        Retrieve a user's open orders.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#retrieve-a-users-open-orders
        """
        return self.info.open_orders(address=self.address)

    def frontend_open_orders(self):
        """Retrieve a user's open orders with additional frontend info.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#retrieve-a-users-open-orders-with-additional-frontend-info
        """
        return self.info.frontend_open_orders(address=self.address)

    def user_fills(self):
        """
        Retrieve a given user's fills.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#retrieve-a-users-fills
        """
        return self.info.user_fills(address=self.address)

    def meta(self):
        """
        Retrieve exchange perpetual metadata.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals#retrieve-perpetuals-metadata
        """
        return self.info.meta()

    def spot_meta(self):
        """
        Retrieve exchange spot metadata.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals#retrieve-perpetuals-metadata
        """
        return self.info.spot_meta()

    def funding_history(
        self,
        coin,
        startms=int((datetime.now(pytz.utc) - timedelta(days=14)).timestamp() * 1000),
        endms=None,
    ):
        """
        Retrieve funding history for a given coin.

        Args:
            coin (str): The coin symbol.
            startms (int, optional): Start time in milliseconds. Defaults to 14 days ago from the current time.
            endms (int, optional): End time in milliseconds. Defaults to None.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals#retrieve-historical-funding-rates
        """
        return self.info.funding_history(coin=coin, startTime=startms, endTime=endms)

    def user_funding_history(
        self,
        startms=int((datetime.now(pytz.utc) - timedelta(days=14)).timestamp() * 1000),
        endms=None,
    ):
        """
        Retrieve a user's funding history.

        Args:
            startms (int, optional): Start time in milliseconds. Defaults to 14 days ago from the current time.
            endms (int, optional): End time in milliseconds. Defaults to None.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals#retrieve-a-users-funding-history
        """
        return self.info.user_funding_history(
            user=self.address, startTime=startms, endTime=endms
        )

    def user_rate_limit(self):
        """
        Retrieve the user's L1 Rate limits.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits
        """
        return self.info.post("/info", {"type": "userRateLimit", "user": self.address})

    def l2_snapshot(self, ticker, nsigfig=None):
        """
        Retrieve L2 snapshot for a given coin.

        Args:
            coin (str): The coin symbol.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#l2-book-snapshot
        """
        return (
            self.info.post("/info", {"type": "l2Book", "coin": ticker})
            if nsigfig is None
            else self.info.post(
                "/info", {"type": "l2Book", "coin": ticker, "nSigFigs": nsigfig}
            )
        )

    def candles_snapshot(self, coin, interval, startms, endms):
        """
        Retrieve candles snapshot for a given coin.

        Args:
            coin (str): The coin symbol.
            interval (str): Candlestick interval.
            startms (int): Start time in milliseconds.
            endms (int): End time in milliseconds.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#candle-snapshot
        """
        return self.info.candles_snapshot(
            coin=coin, interval=interval, startTime=startms, endTime=endms
        )

    def query_order_by_oid(self, oid):
        """
        Retrieve the details for the order using its order id.

        Args:
            oid (str): The order ID.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#query-order-status-by-oid-or-cloid
        """
        return self.info.query_order_by_oid(user=self.address, oid=oid)

    """
    AllMidsSubscription = TypedDict("AllMidsSubscription", {"type": Literal["allMids"]})
    L2BookSubscription = TypedDict("L2BookSubscription", {"type": Literal["l2Book"], "coin": str})
    TradesSubscription = TypedDict("TradesSubscription", {"type": Literal["trades"], "coin": str})
    UserEventsSubscription = TypedDict("UserEventsSubscription", {"type": Literal["userEvents"], "user": str})
    Subscription = Union[AllMidsSubscription, L2BookSubscription, TradesSubscription, UserEventsSubscription]
    
    Async Support for Websocket streaming
    L2 websocket pushes out at the later of 0.5 second or an orderbook change (in a new block)
    """

    def fut_l2_subscriptions(self):
        """
        Retrieve the list of perpetuals with open subscription.
        """
        return self.aws_manager.fut_l2_subscriptions()

    async def fut_l2_subscribe(self, symbol, stream_buffer=None, handler=None):
        """
        Stream l2-order book market data for given coin.

        Args:
            symbol (str): Ticker symbol for the perpetual asset.
            stream_buffer (Union[Deque, defaultdict], Optional): A data structure that stores streaming data. Deque instance contains streamed data. Uses internal buffer if not given.
        """
        if stream_buffer is None:
            stream_buffer = self.stream_buffer
        return await self.aws_manager.fut_l2_subscribe(
            symbol=symbol, stream_buffer=stream_buffer, handler=handler
        )

    async def fut_l2_unsubscribe(self, symbol):
        """
        Terminate stream for coin l2-order book.

        Args:
            symbol (str): Ticker symbol for the perpetual asset.
        """
        return await self.aws_manager.fut_l2_unsubscribe(symbol=symbol)

    def fut_l2_streamed(self, symbol):
        """
        Retrives the streamed l2-order book data from the stream buffer.

        Args:
            symbol (str): Ticker symbol for the perpetual asset.
        """
        return self.aws_manager.fut_l2_streamed(symbol=symbol)

    async def keep_book_state(self, handler=None):
        """
        Keeps a stateful position dictionary using a websocket connection.
        Following calls to `get_book` returns the address's `{ coin : Decimal(coin_pos) }`
        without HTTP requests.
        """
        self.book_manager.init_book()
        subscription = {"type": "userFills", "user": self.address}
        id = subscription_to_identifier(subscription=subscription)

        async def _handler(msg):
            assert msg["channel"] == "userFills"
            if "isSnapshot" in msg["data"] and msg["data"]["isSnapshot"]:
                return
            fills = msg["data"]["fills"]
            for fill in fills:
                self.book_manager.account_fill(fill)
                if self.hook:
                    await self.hook(fill)

        handler = handler if handler is not None else _handler
        self.book_tracked = True
        return await self.aws_manager.stream_subscription(
            id=id, subscription=subscription, handler=handler
        )

    async def get_book(self):
        """
        Returns the address's `{ coin : Decimal(coin_pos) }`.
        Should only be called after `keep_book_state` has been initiated.
        """
        if not self.book_tracked:
            raise Exception("keep_book_state must be initiated first")
        updated_book = defaultdict(lambda: Decimal("0"))
        for pos, value in self.book_manager.book.items():
            if value != Decimal("0"):
                updated_book[pos] = value
        return updated_book

    def get_open_orders(self):
        """
        Retrieve the details of account's all open orders.

        Notes:
            https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint#retrieve-a-users-open-orders
        """
        return self._open_orders()

    async def limit_order(
        self,
        coin,
        amount,
        price,
        reduce_only=False,
        order_type={"limit": {"tif": "Gtc"}},
    ):
        """
        Place a limit order.

        Args:
            coin (str): The coin symbol.
            amount (float): The (positive or negative) amount to trade.
            price (float): The price at which to execute the order.
            reduce_only (bool, optional): Whether the order should reduce an existing position only. Defaults to False.
            order_type (dict, optional): The order type. Defaults to {"limit": {"tif": "Gtc"}}.

        Returns:
            Any: The result of the order placement.
        """
        if amount == 0:
            return
        res = None
        order = {
            "coin": coin,
            "is_buy": (True if amount > 0 else False),
            "sz": float(abs(amount)),
            "limit_px": price,
            "reduce_only": reduce_only,
            "order_type": order_type,
        }
        logging.info(
            f"ORDER INFO", extra={"exchange": "hyp", "type": "ORDER", "submit": order}
        )
        res = await self.exchange.aorder(**order)
        logging.info(
            f"ORDER INFO",
            extra={"exchange": "hyp", "type": "ORDER", "res": res, "submitted": order},
        )
        return res

    async def get_equity(self):
        """
        Retrieve the perpetual account equity of the user.

        Returns:
            float: The equity of the user.
        """
        res = self.user_state()
        equity = float(res["marginSummary"]["accountValue"])
        logging.info(f"retrieve equity {equity}", extra={"exchange": "hyp"})
        return equity

    def is_any_stream_error(self):
        return self.aws_manager.is_any_stream_error()

    async def clean(self):
        """
        Cleans up open sessions with HPL server.
        """
        await self.info.clean()
        await self.exchange.clean()
        await self.book_manager.info.clean()


class BookManager:
    def __init__(self, address):
        self.address = address
        self.info = Info(base_url=constants.MAINNET_API_URL, skip_ws=True)
        self.book = defaultdict(lambda: Decimal("0"))
        self.book_stamp = None

    def init_book(self):
        state = self.info.user_state(address=self.address)
        positions = state["assetPositions"]
        for position in positions:
            self.book[position["position"]["coin"]] += Decimal(
                position["position"]["szi"]
            )
        self.book_stamp = state["time"]
        return self.book

    def account_fill(self, fill):
        time = fill["time"]
        ticker = fill["coin"]
        sz = fill["sz"]
        startpos = fill["startPosition"]
        dir = 1 if fill["side"] == "B" else (-1 if fill["side"] == "A" else 0)
        oid = fill["oid"]
        self.book[ticker] = Decimal(startpos) + Decimal(sz) * Decimal(dir)
        if self.book[ticker] == Decimal("0"):
            del self.book[ticker]


import websockets


def subscription_to_identifier(subscription) -> str:
    if subscription["type"] == "allMids":
        return "allMids"
    elif subscription["type"] == "l2Book":
        return f'l2Book:{subscription["coin"].lower()}'
    elif subscription["type"] == "trades":
        return f'trades:{subscription["coin"].lower()}'
    elif subscription["type"] == "userEvents":
        return "userEvents"
    elif subscription["type"] == "userFills":
        return "userFills"
    elif subscription["type"] == "orderUpdates":
        return "orderUpdates"


class AsyncWebsocketManager:
    def __init__(self, base_url=constants.MAINNET_API_URL, stream_buffer=None):
        self.conns = {}
        self.ws_url = "ws" + base_url[len("http") :] + "/ws"
        self.obj_fut_l2_subscriptions = set()
        self.stream_buffer = (
            stream_buffer
            if stream_buffer is not None
            else defaultdict(lambda: deque(maxlen=100))
        )

    def is_any_stream_error(self):
        return any(isinstance(task, Exception) for id, task in self.conns.items())

    def fut_l2_subscriptions(self):
        logging.info(
            f"retrieving fut_l2_subscriptions: {' '.join(self.obj_fut_l2_subscriptions)}",
            extra={"exchange": "hyp"},
        )
        return self.obj_fut_l2_subscriptions

    async def fut_l2_subscribe(self, symbol, stream_buffer=None, handler=None):
        subscription = {"type": "l2Book", "coin": symbol}
        id = subscription_to_identifier(subscription=subscription)
        self.obj_fut_l2_subscriptions.add(symbol)
        logging.info(f"{id} subscribe", extra={"exchange": "hyp"})

        async def _handler(msg):
            msg_data = msg["data"]
            ts = msg_data["time"]
            bids = msg_data["levels"][0]
            asks = msg_data["levels"][1]
            bids = np.array(
                [[item["px"], item["sz"]] for item in bids], dtype=np.float64
            )
            asks = np.array(
                [[item["px"], item["sz"]] for item in asks], dtype=np.float64
            )
            if stream_buffer:
                stream_buffer[id].append({"ts": ts, "b": bids, "a": asks})

        handler = handler if handler is not None else _handler
        return await self.stream_subscription(
            id=id, subscription=subscription, handler=handler
        )

    async def fut_l2_unsubscribe(self, symbol):
        subscription = {"type": "l2Book", "coin": symbol}
        id = subscription_to_identifier(subscription=subscription)
        logging.info(f"{id} unsubscribe", extra={"exchange": "hyp"})
        self.obj_fut_l2_subscriptions.discard(symbol)
        return await self.end_stream(id=id)

    def fut_l2_streamed(self, symbol):
        subscription = {"type": "l2Book", "coin": symbol}
        id = subscription_to_identifier(subscription=subscription)
        return self.stream_buffer[id]

    async def stream_subscription(self, id, subscription, handler):
        if id in self.conns and not isinstance(self.conns[id], Exception):
            return True
        stream_task = asyncio.create_task(
            self._stream_subscription(id=id, subscription=subscription, handler=handler)
        )
        self.conns[id] = stream_task
        await asyncio.sleep(0)
        return stream_task

    async def _stream_subscription(self, id, subscription, handler):
        stream_socket = websockets.connect(self.ws_url)
        pinger = None
        try:
            async with stream_socket as s:
                await s.send(
                    json.dumps({"method": "subscribe", "subscription": subscription})
                )
                sub = json.loads(await s.recv())
                assert sub["channel"] == "subscriptionResponse"
                pinger = asyncio.create_task(self.ping(s))
                while id in self.conns:
                    msg = json.loads(await s.recv())
                    if msg["channel"] == "pong":
                        continue
                    await handler(msg)

        except Exception as err:
            if id in self.conns:
                self.conns[id] = err
            if pinger is not None:
                pinger.cancel()

    async def ping(self, s):
        try:
            while True:
                await s.send(json.dumps({"method": "ping"}))
                await asyncio.sleep(50)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    async def end_stream(self, id):
        if id not in self.conns:
            return True
        del self.conns[id]
