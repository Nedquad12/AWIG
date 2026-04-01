# =============================================================
# binance_client.py - Binance USDⓈ-M Futures Demo API Client
# =============================================================
# Modul ini handle semua komunikasi dengan Binance Futures API.
# Setiap fungsi return dict hasil response (atau raise Exception).
# =============================================================

import hashlib
import hmac
import time
import urllib.parse

import requests

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_BASE_URL,
    RECV_WINDOW,
)


# ------------------------------------------------------------------
# Helper: Buat signature HMAC-SHA256
# ------------------------------------------------------------------
def _sign(query_string: str) -> str:
    return hmac.new(
        BINANCE_API_SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# ------------------------------------------------------------------
# Helper: Request tanpa signature (public endpoint)
# ------------------------------------------------------------------
def _get_public(path: str, params: dict | None = None) -> dict | list:
    url = BINANCE_BASE_URL + path
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# Helper: Request dengan signature (private endpoint)
# ------------------------------------------------------------------
def _get_signed(path: str, params: dict | None = None) -> dict | list:
    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW

    query_string = urllib.parse.urlencode(params)
    params["signature"] = _sign(query_string)

    url = BINANCE_BASE_URL + path
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# PUBLIC ENDPOINTS
# ------------------------------------------------------------------

def get_server_time() -> int:
    """Return server time dalam milliseconds."""
    data = _get_public("/fapi/v1/time")
    return data["serverTime"]


def get_exchange_info() -> dict:
    """Return info exchange (symbols, filters, limits, dll)."""
    return _get_public("/fapi/v1/exchangeInfo")


def get_ticker_price(symbol: str | None = None) -> dict | list:
    """
    Return harga terakhir.
    - symbol=None  → semua simbol (list)
    - symbol='BTCUSDT' → satu simbol (dict)
    """
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    return _get_public("/fapi/v1/ticker/price", params=params)


def get_24hr_ticker(symbol: str) -> dict:
    """Return statistik 24 jam untuk satu simbol."""
    return _get_public("/fapi/v1/ticker/24hr", params={"symbol": symbol.upper()})


# ------------------------------------------------------------------
# PRIVATE ENDPOINTS - ACCOUNT
# ------------------------------------------------------------------

def get_account_balance() -> list:
    """
    Return list saldo aset di akun futures.
    Tiap item: {asset, balance, crossWalletBalance, availableBalance, ...}
    """
    return _get_signed("/fapi/v2/balance")


def get_account_info() -> dict:
    """
    Return info akun lengkap: total unrealized PnL, margin ratio,
    saldo per aset, dan posisi terbuka.
    """
    return _get_signed("/fapi/v2/account")


# ------------------------------------------------------------------
# PRIVATE ENDPOINTS - POSITIONS
# ------------------------------------------------------------------

def get_position_risk(symbol: str | None = None) -> list:
    """
    Return daftar posisi saat ini (termasuk posisi dengan size 0).
    - symbol=None → semua simbol
    - symbol='BTCUSDT' → filter satu simbol
    """
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    return _get_signed("/fapi/v2/positionRisk", params=params)


def get_open_positions() -> list:
    """Return hanya posisi yang aktif (positionAmt != 0)."""
    all_pos = get_position_risk()
    return [p for p in all_pos if float(p.get("positionAmt", 0)) != 0]


# ------------------------------------------------------------------
# PRIVATE ENDPOINTS - ORDERS
# ------------------------------------------------------------------

def get_open_orders(symbol: str | None = None) -> list:
    """
    Return semua open order.
    - symbol=None → semua simbol (lebih berat, rate limit lebih tinggi)
    - symbol='BTCUSDT' → filter satu simbol
    """
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    return _get_signed("/fapi/v1/openOrders", params=params)


def get_all_orders(symbol: str, limit: int = 10) -> list:
    """
    Return riwayat order (open + filled + cancelled) untuk satu simbol.
    limit: jumlah order terbaru (maks 1000).
    """
    params = {"symbol": symbol.upper(), "limit": limit}
    return _get_signed("/fapi/v1/allOrders", params=params)


def get_order_detail(symbol: str, order_id: int) -> dict:
    """Return detail satu order berdasarkan orderId."""
    params = {"symbol": symbol.upper(), "orderId": order_id}
    return _get_signed("/fapi/v1/order", params=params)


def get_income_history(income_type: str | None = None, limit: int = 20) -> list:
    """
    Return riwayat income (realized PnL, funding, commission, dll).
    income_type contoh: 'REALIZED_PNL', 'FUNDING_FEE', 'COMMISSION'
    """
    params = {"limit": limit}
    if income_type:
        params["incomeType"] = income_type
    return _get_signed("/fapi/v1/income", params=params)
