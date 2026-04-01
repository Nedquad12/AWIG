import hashlib
import hmac
import logging
import math
import os
import sys
import time
import urllib.parse

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_BASE_URL,
    DEFAULT_LEVERAGE,
    RECV_WINDOW,
    RISK_PER_TRADE_PCT,
)

logger = logging.getLogger(__name__)

def _sign(query_string: str) -> str:
    return hmac.new(
        BINANCE_API_SECRET.encode(),
        query_string.encode(),
        hashlib.sha256,
    ).hexdigest()


def _post_signed(path: str, params: dict) -> dict:
    params["timestamp"]  = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW
    query = urllib.parse.urlencode(params)
    params["signature"] = _sign(query)

    url     = BINANCE_BASE_URL + path
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    resp    = requests.post(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_signed(path: str, params: dict) -> dict | list:
    params["timestamp"]  = int(time.time() * 1000)
    params["recvWindow"] = RECV_WINDOW
    query = urllib.parse.urlencode(params)
    params["signature"] = _sign(query)

    url     = BINANCE_BASE_URL + path
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    resp    = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

def _get_symbol_info(symbol: str) -> dict:
    """Return filter LOT_SIZE dan PRICE_FILTER untuk symbol."""
    url  = f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    for s in data.get("symbols", []):
        if s["symbol"] == symbol.upper():
            filters = {f["filterType"]: f for f in s["filters"]}
            return {
                "qty_step":    float(filters["LOT_SIZE"]["stepSize"]),
                "min_qty":     float(filters["LOT_SIZE"]["minQty"]),
                "price_tick":  float(filters["PRICE_FILTER"]["tickSize"]),
            }
    raise ValueError(f"Symbol {symbol} tidak ditemukan di exchange info")


def _round_step(value: float, step: float) -> float:
    """Round value ke kelipatan step (untuk qty & price)."""
    precision = max(0, round(-math.log10(step)))
    factor    = 10 ** precision
    return math.floor(value * factor) / factor

def _get_available_balance() -> float:
    balances = _get_signed("/fapi/v2/balance", {})
    for b in balances:
        if b["asset"] == "USDT":
            return float(b["availableBalance"])
    return 0.0

def execute_order(ai_result: dict, pred: dict) -> dict:
 
    symbol      = pred["symbol"]
    action      = ai_result["action"]          # "BUYING" atau "SELLING"
    entry_price = ai_result["entry_price"]
    stop_loss   = ai_result["stop_loss"]
    take_profit = ai_result["take_profit"]
    leverage    = ai_result["leverage"]

    side = "BUY" if action == "BUYING" else "SELL"

    logger.info("[executor] %s %s @ %.4f (lev=%dx)", side, symbol, entry_price, leverage)

    try:
        # 1. Set leverage
        _post_signed("/fapi/v1/leverage", {
            "symbol":   symbol,
            "leverage": leverage,
        })

        # 2. Hitung quantity
        sym_info  = _get_symbol_info(symbol)
        available = _get_available_balance()
        notional  = available * (RISK_PER_TRADE_PCT / 100) * leverage
        raw_qty   = notional / entry_price
        qty       = _round_step(raw_qty, sym_info["qty_step"])

        if qty < sym_info["min_qty"]:
            msg = (
                f"Quantity {qty} < minimum {sym_info['min_qty']} untuk {symbol}. "
                f"Balance: {available:.2f} USDT"
            )
            logger.error("[executor] %s", msg)
            return {"ok": False, "reason_fail": msg, "symbol": symbol}

        # 3. Round entry price ke tick
        entry_rounded = _round_step(entry_price, sym_info["price_tick"])

        # 4. Place limit order
        order_resp = _post_signed("/fapi/v1/order", {
            "symbol":      symbol,
            "side":        side,
            "type":        "LIMIT",
            "timeInForce": "GTC",
            "quantity":    qty,
            "price":       entry_rounded,
        })

        order_id = order_resp.get("orderId", 0)
        logger.info("[executor] Order placed: id=%s qty=%s price=%s", order_id, qty, entry_rounded)

        return {
            "ok":           True,
            "symbol":       symbol,
            "side":         side,
            "order_id":     order_id,
            "qty":          qty,
            "entry_price":  entry_rounded,
            "stop_loss":    stop_loss,
            "take_profit":  take_profit,
            "leverage":     leverage,
            "balance_used": round(qty * entry_rounded / leverage, 4),
        }

    except requests.HTTPError as e:
        msg = f"Binance HTTP error: {e.response.text if e.response else str(e)}"
        logger.error("[executor] %s", msg)
        return {"ok": False, "reason_fail": msg, "symbol": symbol}

    except Exception as e:
        msg = f"Unexpected error: {e}"
        logger.exception("[executor] %s", msg)
        return {"ok": False, "reason_fail": msg, "symbol": symbol}
