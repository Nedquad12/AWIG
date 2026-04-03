
import hashlib
import hmac
import time
import urllib.parse

import requests

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TRADE_URL,
    RECV_WINDOW,
)


def _sign(query_string: str) -> str:
    return hmac.new(
        BINANCE_API_SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _get_public(path: str, params: dict | None = None) -> dict | list:
    url = BINANCE_TRADE_URL + path
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_signed(path: str, params: dict | None = None) -> dict | list:
    params = params or {}
    params["timestamp"]  = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW

    query_string = urllib.parse.urlencode(params)
    params["signature"] = _sign(query_string)

    url     = BINANCE_TRADE_URL + path
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    resp    = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# PUBLIC ENDPOINTS
# ------------------------------------------------------------------

def get_server_time() -> int:
    data = _get_public("/fapi/v1/time")
    return data["serverTime"]


def get_exchange_info() -> dict:
    return _get_public("/fapi/v1/exchangeInfo")


def get_ticker_price(symbol: str | None = None) -> dict | list:
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    return _get_public("/fapi/v1/ticker/price", params=params)


def get_24hr_ticker(symbol: str) -> dict:
    return _get_public("/fapi/v1/ticker/24hr", params={"symbol": symbol.upper()})


# ------------------------------------------------------------------
# PRIVATE ENDPOINTS - ACCOUNT
# ------------------------------------------------------------------

def get_account_balance() -> list:
    return _get_signed("/fapi/v2/balance")


def get_account_info() -> dict:
    return _get_signed("/fapi/v2/account")


# ------------------------------------------------------------------
# PRIVATE ENDPOINTS - POSITIONS
# ------------------------------------------------------------------

def get_position_risk(symbol: str | None = None) -> list:
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    return _get_signed("/fapi/v2/positionRisk", params=params)


def get_open_positions() -> list:
    all_pos = get_position_risk()
    return [p for p in all_pos if float(p.get("positionAmt", 0)) != 0]


# ------------------------------------------------------------------
# PRIVATE ENDPOINTS - ORDERS
# ------------------------------------------------------------------

def get_open_orders(symbol: str | None = None) -> list:
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    return _get_signed("/fapi/v1/openOrders", params=params)


def get_all_orders(symbol: str, limit: int = 10) -> list:
    params = {"symbol": symbol.upper(), "limit": limit}
    return _get_signed("/fapi/v1/allOrders", params=params)


def get_order_detail(symbol: str, order_id: int) -> dict:
    params = {"symbol": symbol.upper(), "orderId": order_id}
    return _get_signed("/fapi/v1/order", params=params)


def get_income_history(income_type: str | None = None, limit: int = 20) -> list:
    params = {"limit": limit}
    if income_type:
        params["incomeType"] = income_type
    return _get_signed("/fapi/v1/income", params=params)
