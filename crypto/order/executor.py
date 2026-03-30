"""
order/executor.py — Eksekusi entry LIMIT, tunggu fill di background, pasang SL + TP.

Polling berjalan di daemon thread terpisah — tidak memblok scheduler,
tidak memblok request user Telegram, dan tidak mengganggu pipeline token lain.

Alur:
  1. Set leverage
  2. Place LIMIT entry → return segera ke pipeline (non-blocking)
  3. Background thread poll setiap 3 detik, max 20 menit
  4. Jika FILLED → pasang SL + TP (reduceOnly + quantity eksplisit)
  5. Jika timeout / cancel → log + notify tanpa mengganggu apapun
  6. Jika SL/TP gagal setelah fill → emergency market close
"""

import hashlib
import hmac
import logging
import math
import os
import sys
import threading
import time
import urllib.parse

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    BINANCE_TRADE_URL,
    RECV_WINDOW,
    RISK_PER_TRADE_PCT,
)

logger = logging.getLogger(__name__)

FILL_TIMEOUT_SEC   = 20 * 60   # 20 menit
FILL_POLL_INTERVAL = 3          # cek tiap 3 detik


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def _sign(qs: str) -> str:
    return hmac.new(
        BINANCE_API_SECRET.encode(),
        qs.encode(),
        hashlib.sha256,
    ).hexdigest()


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


def _get(path: str, params: dict) -> dict | list:
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


# ------------------------------------------------------------------
# Symbol info & rounding
# ------------------------------------------------------------------

def _get_symbol_info(symbol: str) -> dict:
    url  = f"{BINANCE_TRADE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    for s in resp.json().get("symbols", []):
        if s["symbol"] == symbol.upper():
            filters = {f["filterType"]: f for f in s["filters"]}
            return {
                "qty_step":   float(filters["LOT_SIZE"]["stepSize"]),
                "min_qty":    float(filters["LOT_SIZE"]["minQty"]),
                "price_tick": float(filters["PRICE_FILTER"]["tickSize"]),
            }
    raise ValueError(f"Symbol {symbol} tidak ditemukan di exchange info")


def _round_step(value: float, step: float) -> float:
    precision = max(0, round(-math.log10(step)))
    return math.floor(value * 10**precision) / 10**precision


def _round_price(value: float, tick: float) -> float:
    precision = max(0, round(-math.log10(tick)))
    return round(value, precision)


def _get_available_balance() -> float:
    for b in _get("/fapi/v2/balance", {}):
        if b["asset"] == "USDT":
            return float(b["availableBalance"])
    return 0.0


# ------------------------------------------------------------------
# Emergency close
# ------------------------------------------------------------------

def _cancel_order(symbol: str, order_id: int) -> None:
    try:
        _delete("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
        logger.info("[executor] Order %d di-cancel", order_id)
    except Exception as e:
        logger.error("[executor] Gagal cancel order %d: %s", order_id, e)


def _emergency_close(symbol: str, side: str, qty: float) -> None:
    close_side = "SELL" if side == "BUY" else "BUY"
    try:
        _post("/fapi/v1/order", {
            "symbol":     symbol,
            "side":       close_side,
            "type":       "MARKET",
            "quantity":   qty,
            "reduceOnly": "true",
        })
        logger.warning("[executor] Emergency MARKET close %s qty=%s", symbol, qty)
    except Exception as e:
        logger.error("[executor] Emergency close GAGAL %s: %s — POSISI TERBUKA!", symbol, e)


# ------------------------------------------------------------------
# Background worker: poll fill → pasang SL/TP
# ------------------------------------------------------------------

def _sltp_worker(
    symbol: str,
    order_id: int,
    side: str,
    sl_side: str,
    tp_side: str,
    qty: float,
    sl_r: float,
    tp_r: float,
    notify_fn,           # callable(str) atau None
) -> None:
    """
    Daemon thread: poll order sampai FILLED, lalu pasang SL + TP.
    Berjalan sepenuhnya di background — tidak memblok apapun.
    """
    logger.info(
        "[executor:bg] Start polling order %d %s (max %ds, interval %ds)",
        order_id, symbol, FILL_TIMEOUT_SEC, FILL_POLL_INTERVAL,
    )

    def _notify(msg: str) -> None:
        if notify_fn:
            try:
                notify_fn(msg)
            except Exception as e:
                logger.debug("[executor:bg] notify error: %s", e)

    deadline = time.time() + FILL_TIMEOUT_SEC
    filled_qty   = qty
    filled_price = 0.0

    # ── Poll sampai fill ─────────────────────────────────────────
    while time.time() < deadline:
        time.sleep(FILL_POLL_INTERVAL)
        try:
            order  = _get("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
            status = order.get("status", "")

            if status == "FILLED":
                filled_qty   = float(order.get("executedQty", qty))
                filled_price = float(order.get("avgPrice", 0))
                logger.info(
                    "[executor:bg] Order %d FILLED qty=%s avgPrice=%s",
                    order_id, filled_qty, filled_price,
                )
                _notify(
                    f"✅ <b>Entry FILLED — {symbol}</b>\n"
                    f"  Order ID : <code>{order_id}</code>\n"
                    f"  Side     : <b>{side}</b>\n"
                    f"  Qty      : <code>{filled_qty}</code>\n"
                    f"  Avg Price: <code>{filled_price}</code>\n"
                    f"  Memasang SL & TP..."
                )
                break

            if status in ("CANCELED", "EXPIRED", "REJECTED"):
                logger.warning(
                    "[executor:bg] Order %d terminal status: %s — stop polling",
                    order_id, status,
                )
                _notify(
                    f"⚠️ <b>{symbol}</b> — Entry order {order_id} "
                    f"berakhir dengan status <b>{status}</b>, SL/TP tidak dipasang."
                )
                return

            logger.debug("[executor:bg] Order %d status=%s, lanjut poll...", order_id, status)

        except Exception as e:
            logger.warning("[executor:bg] Poll error order %d: %s", order_id, e)
            # Lanjut poll, jangan hentikan karena error sementara

    else:
        # Timeout — cancel entry
        logger.warning(
            "[executor:bg] Order %d timeout setelah %ds — cancel",
            order_id, FILL_TIMEOUT_SEC,
        )
        _cancel_order(symbol, order_id)
        _notify(
            f"⏱ <b>{symbol}</b> — Entry order {order_id} tidak ter-fill "
            f"dalam {FILL_TIMEOUT_SEC//60} menit, di-cancel."
        )
        return

    # ── Pasang SL ────────────────────────────────────────────────
    sl_order_id = None
    try:
        sl_resp = _post("/fapi/v1/order", {
            "symbol":       symbol,
            "side":         sl_side,
            "type":         "STOP_MARKET",
            "stopPrice":    sl_r,
            "quantity":     filled_qty,
            "reduceOnly":   "true",
            "workingType":  "MARK_PRICE",
            "priceProtect": "true",
        })
        sl_order_id = sl_resp.get("orderId", 0)
        logger.info("[executor:bg] SL placed: id=%d stopPrice=%s", sl_order_id, sl_r)
    except Exception as e:
        logger.error("[executor:bg] SL gagal: %s — emergency close", e)
        _emergency_close(symbol, side, filled_qty)
        _notify(
            f"🚨 <b>{symbol}</b> — SL gagal dipasang: <code>{e}</code>\n"
            f"Posisi di-close via market order."
        )
        return

    # ── Pasang TP ────────────────────────────────────────────────
    try:
        tp_resp = _post("/fapi/v1/order", {
            "symbol":       symbol,
            "side":         tp_side,
            "type":         "TAKE_PROFIT_MARKET",
            "stopPrice":    tp_r,
            "quantity":     filled_qty,
            "reduceOnly":   "true",
            "workingType":  "MARK_PRICE",
            "priceProtect": "true",
        })
        tp_order_id = tp_resp.get("orderId", 0)
        logger.info("[executor:bg] TP placed: id=%d stopPrice=%s", tp_order_id, tp_r)
    except Exception as e:
        logger.error("[executor:bg] TP gagal: %s — cancel SL + emergency close", e)
        if sl_order_id:
            _cancel_order(symbol, sl_order_id)
        _emergency_close(symbol, side, filled_qty)
        _notify(
            f"🚨 <b>{symbol}</b> — TP gagal dipasang: <code>{e}</code>\n"
            f"SL di-cancel, posisi di-close via market order."
        )
        return

    # ── Sukses ───────────────────────────────────────────────────
    side_emoji = "🟢" if side == "BUY" else "🔴"
    _notify(
        f"{side_emoji} <b>SL & TP Aktif — {symbol}</b>\n"
        f"  SL Order ID : <code>{sl_order_id}</code>  @ <code>{sl_r}</code> ✅\n"
        f"  TP Order ID : <code>{tp_order_id}</code>  @ <code>{tp_r}</code> ✅"
    )
    logger.info(
        "[executor:bg] %s SL=%d TP=%d — semua order aktif",
        symbol, sl_order_id, tp_order_id,
    )


# ------------------------------------------------------------------
# Main: execute_order
# ------------------------------------------------------------------

def execute_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """
    Place entry LIMIT lalu jalankan background thread untuk poll + SL/TP.
    Return segera setelah entry order dikirim — tidak memblok.

    Args:
        notify_fn : callable(str) opsional untuk kirim update via Telegram
                    saat entry fill atau SL/TP terpasang.
    """
    symbol       = pred["symbol"]
    action       = ai_result["action"]
    entry_price  = float(ai_result["entry_price"])
    stop_loss    = float(ai_result["stop_loss"])
    take_profit  = float(ai_result["take_profit"])
    leverage     = int(ai_result["leverage"])
    qty_fraction = float(ai_result.get("qty_fraction", RISK_PER_TRADE_PCT / 100))
    qty_fraction = max(0.001, min(qty_fraction, 1.0))

    side    = "BUY"  if action == "BUYING" else "SELL"
    sl_side = "SELL" if side == "BUY"      else "BUY"
    tp_side = "SELL" if side == "BUY"      else "BUY"

    logger.info(
        "[executor] %s %s entry=%.6f SL=%.6f TP=%.6f lev=%dx fraction=%.4f",
        side, symbol, entry_price, stop_loss, take_profit, leverage, qty_fraction,
    )

    # Validasi arah SL/TP
    if side == "BUY":
        if stop_loss >= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"SL {stop_loss} >= entry {entry_price} untuk LONG"}
        if take_profit <= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"TP {take_profit} <= entry {entry_price} untuk LONG"}
    else:
        if stop_loss <= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"SL {stop_loss} <= entry {entry_price} untuk SHORT"}
        if take_profit >= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"TP {take_profit} >= entry {entry_price} untuk SHORT"}

    try:
        # 1. Set leverage
        _post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

        # 2. Hitung qty
        sym_info  = _get_symbol_info(symbol)
        available = _get_available_balance()
        notional  = available * qty_fraction * leverage
        qty       = _round_step(notional / entry_price, sym_info["qty_step"])

        if qty < sym_info["min_qty"]:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": (
                        f"Qty {qty} < min {sym_info['min_qty']}. "
                        f"Balance={available:.2f} fraction={qty_fraction:.4f}"
                    )}

        entry_r = _round_price(entry_price, sym_info["price_tick"])
        sl_r    = _round_price(stop_loss,   sym_info["price_tick"])
        tp_r    = _round_price(take_profit,  sym_info["price_tick"])

        # 3. Place LIMIT entry
        entry_resp = _post("/fapi/v1/order", {
            "symbol":      symbol,
            "side":        side,
            "type":        "LIMIT",
            "timeInForce": "GTC",
            "quantity":    qty,
            "price":       entry_r,
        })
        order_id = entry_resp.get("orderId", 0)
        logger.info("[executor] Entry LIMIT placed: id=%d qty=%s @ %s", order_id, qty, entry_r)

        # 4. Jalankan background thread untuk poll + SL/TP
        t = threading.Thread(
            target=_sltp_worker,
            args=(symbol, order_id, side, sl_side, tp_side, qty, sl_r, tp_r, notify_fn),
            daemon=True,   # mati otomatis saat main process shutdown
            name=f"sltp-{symbol}-{order_id}",
        )
        t.start()
        logger.info("[executor] Background thread started: %s", t.name)

        # Return segera — pipeline bisa lanjut ke token berikutnya
        return {
            "ok":           True,
            "symbol":       symbol,
            "side":         side,
            "order_id":     order_id,
            "sl_order_id":  None,   # belum ada, dipasang di background
            "tp_order_id":  None,
            "qty":          qty,
            "entry_price":  entry_r,
            "stop_loss":    sl_r,
            "take_profit":  tp_r,
            "leverage":     leverage,
            "balance_used": round(qty * entry_r / leverage, 4),
            "qty_fraction": round(qty_fraction, 6),
            "note":         f"SL/TP dipasang otomatis setelah entry fill (max {FILL_TIMEOUT_SEC//60} menit)",
        }

    except requests.HTTPError as e:
        msg = f"Binance HTTP error: {e.response.text if e.response else str(e)}"
        logger.error("[executor] %s", msg)
        return {"ok": False, "reason_fail": msg, "symbol": symbol}
    except Exception as e:
        msg = f"Unexpected error: {e}"
        logger.exception("[executor] %s", msg)
        return {"ok": False, "reason_fail": msg, "symbol": symbol}
