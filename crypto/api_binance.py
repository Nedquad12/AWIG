"""
api_binance.py — Fetch data OHLCV 15m dari Binance Futures Testnet

Endpoint: GET /fapi/v1/klines
"""

import hashlib
import hmac
import logging
import time
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    BINANCE_BASE_URL, CANDLE_LIMIT, TIMEFRAME,
)

logger = logging.getLogger(__name__)


def _sign(params: dict) -> str:
    """HMAC SHA256 signature untuk authenticated endpoints."""
    query = urlencode(params)
    return hmac.new(
        BINANCE_API_SECRET.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _headers() -> dict:
    return {"X-MBX-APIKEY": BINANCE_API_KEY}


def fetch_ohlcv(
    symbol: str,
    interval: str = TIMEFRAME,
    limit: int = CANDLE_LIMIT,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV candles dari Binance Futures Testnet.

    Args:
        symbol   : e.g. "BTCUSDT"
        interval : "15m", "1h", dll
        limit    : jumlah candle (max 1500)

    Returns:
        DataFrame dengan kolom: date, open, high, low, close, volume, transactions
        None jika gagal
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/klines"
    params = {
        "symbol":   symbol.upper(),
        "interval": interval,
        "limit":    limit,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            logger.warning(f"[{symbol}] Tidak ada data dari Binance")
            return None

        # Binance klines format:
        # [open_time, open, high, low, close, volume, close_time,
        #  quote_vol, num_trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]
        rows = []
        for k in data:
            rows.append({
                "timestamp":    int(k[0]),
                "open":         float(k[1]),
                "high":         float(k[2]),
                "low":          float(k[3]),
                "close":        float(k[4]),
                "volume":       float(k[5]),
                "transactions": int(k[8]),   # num_trades
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(
            f"[{symbol}] {len(df)} candle {interval}, "
            f"terakhir: {df['date'].iloc[-1]}"
        )
        return df

    except requests.exceptions.HTTPError as e:
        logger.error(f"[{symbol}] HTTP error: {e} — {resp.text[:200]}")
    except requests.exceptions.Timeout:
        logger.error(f"[{symbol}] Request timeout")
    except Exception as e:
        logger.error(f"[{symbol}] Error: {e}")

    return None


def get_account_balance(asset: str = "USDT") -> float:
    """
    Ambil balance akun Futures Testnet.

    Returns:
        float balance, 0.0 jika gagal
    """
    url = f"{BINANCE_BASE_URL}/fapi/v2/account"
    ts  = int(time.time() * 1000)
    params = {"timestamp": ts}
    params["signature"] = _sign(params)

    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for a in data.get("assets", []):
            if a["asset"] == asset:
                return float(a["availableBalance"])
        return 0.0

    except Exception as e:
        logger.error(f"Gagal get balance: {e}")
        return 0.0


def get_mark_price(symbol: str) -> float:
    """Ambil mark price terbaru."""
    url = f"{BINANCE_BASE_URL}/fapi/v1/premiumIndex"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        resp.raise_for_status()
        return float(resp.json()["markPrice"])
    except Exception as e:
        logger.error(f"[{symbol}] Gagal get mark price: {e}")
        return 0.0


def get_open_positions() -> list[dict]:
    """
    Ambil semua posisi yang sedang terbuka.

    Returns:
        list of dict {symbol, positionAmt, entryPrice, unrealizedProfit, leverage}
    """
    url = f"{BINANCE_BASE_URL}/fapi/v2/positionRisk"
    ts  = int(time.time() * 1000)
    params = {"timestamp": ts}
    params["signature"] = _sign(params)

    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()

        positions = []
        for p in data:
            amt = float(p["positionAmt"])
            if amt != 0:
                positions.append({
                    "symbol":           p["symbol"],
                    "positionAmt":      amt,
                    "entryPrice":       float(p["entryPrice"]),
                    "unrealizedProfit": float(p["unRealizedProfit"]),
                    "leverage":         int(p["leverage"]),
                    "side":             "LONG" if amt > 0 else "SHORT",
                })
        return positions

    except Exception as e:
        logger.error(f"Gagal get positions: {e}")
        return []


def set_leverage(symbol: str, leverage: int) -> bool:
    """Set leverage untuk simbol tertentu."""
    url = f"{BINANCE_BASE_URL}/fapi/v1/leverage"
    ts  = int(time.time() * 1000)
    params = {
        "symbol":    symbol.upper(),
        "leverage":  leverage,
        "timestamp": ts,
    }
    params["signature"] = _sign(params)

    try:
        resp = requests.post(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        logger.info(f"[{symbol}] Leverage di-set ke {leverage}x")
        return True
    except Exception as e:
        logger.error(f"[{symbol}] Gagal set leverage: {e}")
        return False


def place_market_order(
    symbol:   str,
    side:     str,   # "BUY" atau "SELL"
    quantity: float,
) -> Optional[dict]:
    """
    Place market order di Futures Testnet.

    Args:
        symbol   : e.g. "BTCUSDT"
        side     : "BUY" (LONG) atau "SELL" (SHORT)
        quantity : jumlah kontrak (dalam satuan base asset)

    Returns:
        dict response dari Binance, None jika gagal
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/order"
    ts  = int(time.time() * 1000)
    params = {
        "symbol":     symbol.upper(),
        "side":       side.upper(),
        "type":       "MARKET",
        "quantity":   quantity,
        "timestamp":  ts,
    }
    params["signature"] = _sign(params)

    try:
        resp = requests.post(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        order = resp.json()
        logger.info(
            f"[{symbol}] Order {side} {quantity} → "
            f"orderId={order.get('orderId')} status={order.get('status')}"
        )
        return order
    except Exception as e:
        body = ""
        try:
            body = resp.text[:300]
        except Exception:
            pass
        logger.error(f"[{symbol}] Gagal place order: {e} | {body}")
        return None


def close_position(symbol: str, position_amt: float) -> Optional[dict]:
    """
    Tutup posisi yang sedang terbuka dengan market order berlawanan.

    Args:
        symbol       : e.g. "BTCUSDT"
        position_amt : nilai dari positionAmt (positif = LONG, negatif = SHORT)
    """
    if position_amt > 0:
        side = "SELL"
        qty  = abs(position_amt)
    else:
        side = "BUY"
        qty  = abs(position_amt)

    return place_market_order(symbol, side, qty)


def get_exchange_info(symbol: str) -> dict:
    """
    Ambil info trading rules untuk simbol (step size, min qty, dll).
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for s in data.get("symbols", []):
            if s["symbol"] == symbol.upper():
                return s
        return {}
    except Exception as e:
        logger.error(f"Gagal get exchange info: {e}")
        return {}


def place_stop_order(
    symbol:     str,
    side:       str,    # "BUY" (untuk close SHORT) atau "SELL" (untuk close LONG)
    quantity:   float,
    stop_price: float,
    order_type: str = "STOP_MARKET",   # STOP_MARKET atau TAKE_PROFIT_MARKET
) -> Optional[dict]:
    """
    Place stop loss atau take profit order (STOP_MARKET / TAKE_PROFIT_MARKET).

    Args:
        symbol     : e.g. "BTCUSDT"
        side       : "SELL" untuk LONG position, "BUY" untuk SHORT position
        quantity   : jumlah kontrak (sama dengan posisi entry)
        stop_price : harga trigger
        order_type : "STOP_MARKET" untuk SL, "TAKE_PROFIT_MARKET" untuk TP

    Returns:
        dict response dari Binance, None jika gagal
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/order"
    ts  = int(time.time() * 1000)

    # Round stop price ke presisi yang wajar
    stop_price_str = f"{stop_price:.4f}"

    params = {
        "symbol":           symbol.upper(),
        "side":             side.upper(),
        "type":             order_type,
        "stopPrice":        stop_price_str,
        "quantity":         quantity,
        "closePosition":    "false",
        "workingType":      "MARK_PRICE",   # trigger berdasarkan mark price
        "priceProtect":     "true",
        "timestamp":        ts,
    }
    params["signature"] = _sign(params)

    try:
        resp = requests.post(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        order = resp.json()
        logger.info(
            f"[{symbol}] {order_type} {side} @ {stop_price_str} → "
            f"orderId={order.get('orderId')} status={order.get('status')}"
        )
        return order
    except Exception as e:
        body = ""
        try:
            body = resp.text[:300]
        except Exception:
            pass
        logger.error(f"[{symbol}] Gagal place {order_type}: {e} | {body}")
        return None


def cancel_open_orders(symbol: str) -> bool:
    """
    Batalkan semua open order (SL/TP) untuk simbol tertentu.
    Dipanggil saat posisi ditutup manual supaya SL/TP tidak menggantung.
    """
    url = f"{BINANCE_BASE_URL}/fapi/v1/allOpenOrders"
    ts  = int(time.time() * 1000)
    params = {"symbol": symbol.upper(), "timestamp": ts}
    params["signature"] = _sign(params)

    try:
        resp = requests.delete(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"[{symbol}] Cancelled open orders: {data}")
        return True
    except Exception as e:
        logger.error(f"[{symbol}] Gagal cancel orders: {e}")
        return False


def get_open_orders(symbol: str) -> list[dict]:
    """Ambil semua open order (SL/TP pending) untuk simbol."""
    url = f"{BINANCE_BASE_URL}/fapi/v1/openOrders"
    ts  = int(time.time() * 1000)
    params = {"symbol": symbol.upper(), "timestamp": ts}
    params["signature"] = _sign(params)

    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[{symbol}] Gagal get open orders: {e}")
        return []
