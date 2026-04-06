"""
order/executor.py — Pure bridge ke Binance Futures API.

Tanggung jawab:
  - Kirim / cancel order ke Binance (LIMIT, MARKET, STOP_MARKET, TAKE_PROFIT_MARKET)
  - Set leverage + clamp ke max yang diizinkan Binance
  - Ambil balance, symbol info, mark price dari Binance
  - TIDAK ada logika bisnis — semua keputusan ada di paper_executor.py

Dipanggil oleh paper_executor.py saat PAPER_TRADING_MODE = False.
"""

import hashlib
import hmac
import logging
import math
import os
import sys
import time
import urllib.parse
from decimal import Decimal, ROUND_DOWN

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TRADE_URL,
    RECV_WINDOW,
)

logger = logging.getLogger(__name__)

MAX_NOTIONAL_USDT = 500.0

# ── Cache balance (update tiap 35 detik) ─────────────────────────────────────
_balance_cache: dict = {"value": None, "ts": 0.0}
_BALANCE_CACHE_TTL = 35.0


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sign(qs: str) -> str:
    return hmac.new(BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()


def _headers() -> dict:
    return {"X-MBX-APIKEY": BINANCE_API_KEY}


def _post(path: str, params: dict) -> dict:
    params["timestamp"]  = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW
    qs = urllib.parse.urlencode(params)
    params["signature"] = _sign(qs)
    resp = requests.post(BINANCE_TRADE_URL + path, params=params,
                         headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, params: dict = None) -> dict | list:
    params = params or {}
    params["timestamp"]  = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW
    qs = urllib.parse.urlencode(params)
    params["signature"] = _sign(qs)
    resp = requests.get(BINANCE_TRADE_URL + path, params=params,
                        headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str, params: dict) -> dict:
    params["timestamp"]  = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW
    qs = urllib.parse.urlencode(params)
    params["signature"] = _sign(qs)
    resp = requests.delete(BINANCE_TRADE_URL + path, params=params,
                           headers=_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Market data helpers (dipanggil paper_executor juga)
# ─────────────────────────────────────────────────────────────────────────────

def get_available_balance() -> float:
    """
    Ambil available USDT balance dari Binance.
    Cache 35 detik agar tidak spam API.
    """
    now = time.time()
    if _balance_cache["value"] is not None and (now - _balance_cache["ts"]) < _BALANCE_CACHE_TTL:
        return _balance_cache["value"]

    try:
        balances = _get("/fapi/v2/balance")
        for b in balances:
            if b["asset"] == "USDT":
                val = float(b["availableBalance"])
                _balance_cache["value"] = val
                _balance_cache["ts"]    = now
                logger.info("[executor] Balance refreshed: %.2f USDT", val)
                return val
    except Exception as e:
        logger.warning("[executor] Gagal fetch balance: %s — pakai cache lama", e)
        if _balance_cache["value"] is not None:
            return _balance_cache["value"]
    return 0.0


def invalidate_balance_cache() -> None:
    """Force refresh balance pada pemanggilan berikutnya."""
    _balance_cache["ts"] = 0.0


def get_symbol_info(symbol: str) -> dict:
    """
    Ambil filter LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL dari exchange info.
    Return dict: qty_step, min_qty, max_qty, price_tick, min_notional.
    """
    url  = f"{BINANCE_TRADE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    for s in resp.json().get("symbols", []):
        if s["symbol"] == symbol.upper():
            filters = {f["filterType"]: f for f in s["filters"]}
            lot      = filters.get("LOT_SIZE", {})
            price    = filters.get("PRICE_FILTER", {})
            notional = filters.get("MIN_NOTIONAL", {})
            info = {
                "qty_step":     float(lot.get("stepSize",  "0.001")),
                "min_qty":      float(lot.get("minQty",    "0.001")),
                "max_qty":      float(lot.get("maxQty",    "999999999")),
                "price_tick":   float(price.get("tickSize", "0.0001")),
                "min_notional": float(notional.get("notional", "5")),
            }
            logger.debug("[executor] %s symbol_info=%s", symbol, info)
            return info

    raise ValueError(f"Symbol {symbol} tidak ditemukan di exchangeInfo")


def get_max_leverage(symbol: str) -> int:
    """
    Cek leverage bracket Binance untuk symbol.
    Return max leverage yang diizinkan (bracket notional terkecil = lev tertinggi).
    Fallback 125 jika gagal agar tidak salah clamp.
    """
    try:
        data = _get("/fapi/v1/leverageBracket", {"symbol": symbol})
        brackets_list = data if isinstance(data, list) else [data]
        for item in brackets_list:
            if item.get("symbol") == symbol:
                brackets = item.get("brackets", [])
                if brackets:
                    max_lev = int(brackets[0].get("initialLeverage", 1))
                    logger.info("[executor] %s max leverage Binance: %dx", symbol, max_lev)
                    return max_lev
    except Exception as e:
        logger.warning("[executor] Gagal ambil leverage bracket %s: %s", symbol, e)
    return 125


def get_mark_price(symbol: str) -> float | None:
    """Ambil mark price terkini dari Binance."""
    try:
        data = requests.get(
            f"{BINANCE_TRADE_URL}/fapi/v1/premiumIndex",
            params={"symbol": symbol}, timeout=5,
        ).json()
        return float(data.get("markPrice", 0)) or None
    except Exception as e:
        logger.warning("[executor] Gagal mark price %s: %s", symbol, e)
        return None


def get_tick_size(symbol: str) -> float:
    """Ambil tickSize untuk symbol (dipakai monitor trailing stop)."""
    try:
        return get_symbol_info(symbol)["price_tick"]
    except Exception:
        return 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Rounding helpers (dipakai paper_executor juga)
# ─────────────────────────────────────────────────────────────────────────────

def round_step(value: float, step: float) -> float | int:
    if step >= 1.0:
        return int(math.floor(value / step) * step)
    precision = max(0, round(-math.log10(step)))
    return math.floor(value * 10**precision) / 10**precision


def round_price(value: float, tick: float) -> float:
    tick_dec = Decimal(str(tick))
    val_dec  = Decimal(str(value))
    return float(val_dec.quantize(tick_dec, rounding=ROUND_DOWN))


# ─────────────────────────────────────────────────────────────────────────────
# Order actions — dipanggil paper_executor saat live mode
# ─────────────────────────────────────────────────────────────────────────────

def set_leverage(symbol: str, leverage: int) -> int:
    """
    Set leverage ke Binance, sudah di-clamp ke max yang diizinkan.
    Return leverage final yang dipakai.
    """
    max_lev = get_max_leverage(symbol)
    if leverage > max_lev:
        logger.info("[executor] %s leverage %dx > max %dx — clamp ke %dx",
                    symbol, leverage, max_lev, max_lev)
        leverage = max_lev
    _post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})
    return leverage


def place_limit_order(symbol: str, side: str, qty: float, price: float) -> dict:
    """Kirim LIMIT GTC order. Return response Binance."""
    resp = _post("/fapi/v1/order", {
        "symbol":      symbol,
        "side":        side,
        "type":        "LIMIT",
        "timeInForce": "GTC",
        "quantity":    qty,
        "price":       price,
    })
    logger.info("[executor] LIMIT placed %s %s qty=%s @ %s id=%s",
                side, symbol, qty, price, resp.get("orderId"))
    return resp


def place_market_order(symbol: str, side: str, qty: float,
                       reduce_only: bool = False) -> dict:
    """Kirim MARKET order. reduce_only=True untuk close posisi."""
    params = {
        "symbol":   symbol,
        "side":     side,
        "type":     "MARKET",
        "quantity": qty,
    }
    if reduce_only:
        params["reduceOnly"] = "true"
    resp = _post("/fapi/v1/order", params)
    logger.info("[executor] MARKET %s %s qty=%s reduceOnly=%s id=%s",
                side, symbol, qty, reduce_only, resp.get("orderId"))
    return resp


def place_stop_market(symbol: str, side: str, stop_price: float,
                      close_position: bool = True) -> dict:
    """Kirim STOP_MARKET order (SL). close_position=True → closePosition."""
    resp = _post("/fapi/v1/order", {
        "symbol":        symbol,
        "side":          side,
        "type":          "STOP_MARKET",
        "stopPrice":     stop_price,
        "closePosition": "true" if close_position else "false",
        "workingType":   "MARK_PRICE",
    })
    logger.info("[executor] STOP_MARKET %s %s stopPrice=%s id=%s",
                side, symbol, stop_price, resp.get("orderId"))
    return resp


def place_take_profit_market(symbol: str, side: str, stop_price: float,
                              close_position: bool = True) -> dict:
    """Kirim TAKE_PROFIT_MARKET order (TP)."""
    resp = _post("/fapi/v1/order", {
        "symbol":        symbol,
        "side":          side,
        "type":          "TAKE_PROFIT_MARKET",
        "stopPrice":     stop_price,
        "closePosition": "true" if close_position else "false",
        "workingType":   "MARK_PRICE",
    })
    logger.info("[executor] TP_MARKET %s %s stopPrice=%s id=%s",
                side, symbol, stop_price, resp.get("orderId"))
    return resp


def cancel_order(symbol: str, order_id: int) -> None:
    """Cancel satu order by ID."""
    try:
        _delete("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
        logger.info("[executor] Cancelled order %d %s", order_id, symbol)
    except Exception as e:
        logger.warning("[executor] Gagal cancel order %d %s: %s", order_id, symbol, e)


def cancel_all_open_orders(symbol: str) -> None:
    """Cancel semua open order untuk symbol (dipakai saat volume reversal / CB)."""
    try:
        _delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
        logger.info("[executor] Cancelled all open orders %s", symbol)
    except Exception as e:
        logger.warning("[executor] Gagal cancel all orders %s: %s", symbol, e)


def get_order_status(symbol: str, order_id: int) -> dict:
    """Poll status satu order."""
    return _get("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})


def amend_sl_to_price(symbol: str, old_sl_order_id: int,
                       side: str, new_stop_price: float) -> int | None:
    """
    Pindahkan SL Binance ke harga baru (cancel lama → pasang baru).
    Return order_id baru atau None jika gagal.
    Dipakai monitor saat breakeven hit → pindah SL ke entry.
    """
    cancel_order(symbol, old_sl_order_id)
    sl_side = "SELL" if side == "BUY" else "BUY"
    try:
        resp = place_stop_market(symbol, sl_side, new_stop_price, close_position=True)
        new_id = resp.get("orderId")
        logger.info("[executor] SL amended %s → %.6f new_id=%s", symbol, new_stop_price, new_id)
        return new_id
    except Exception as e:
        logger.error("[executor] Gagal pasang SL baru %s @ %.6f: %s", symbol, new_stop_price, e)
        return None


def close_position_market(symbol: str, side: str, qty: float) -> dict:
    """
    Close posisi via MARKET order reduce-only.
    side = side posisi yang mau ditutup (bukan close side).
    """
    close_side = "SELL" if side == "BUY" else "BUY"
    try:
        resp = place_market_order(symbol, close_side, qty, reduce_only=True)
        logger.info("[executor] Position closed %s qty=%s", symbol, qty)
        return resp
    except Exception as e:
        logger.error("[executor] Gagal close %s: %s — POSISI MUNGKIN MASIH TERBUKA!", symbol, e)
        raise


def poll_until_filled(symbol: str, order_id: int,
                      timeout_sec: int = 1200,
                      poll_interval: int = 3) -> dict | None:
    """
    Poll order sampai FILLED atau timeout.
    Return order dict saat filled, None saat timeout/cancel/reject.
    Dipanggil di background thread oleh paper_executor.
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(poll_interval)
        try:
            order = get_order_status(symbol, order_id)
            status = order.get("status", "")
            if status == "FILLED":
                logger.info("[executor] Order %d FILLED @ %s", order_id,
                            order.get("avgPrice"))
                return order
            if status in ("CANCELED", "EXPIRED", "REJECTED"):
                logger.warning("[executor] Order %d %s", order_id, status)
                return None
        except Exception as e:
            logger.warning("[executor] Poll error order %d: %s", order_id, e)

    logger.warning("[executor] Order %d timeout %ds — cancel", order_id, timeout_sec)
    cancel_order(symbol, order_id)
    return None
