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

FILL_TIMEOUT_SEC   = 20 * 60   
FILL_POLL_INTERVAL = 3         

MAX_NOTIONAL_USDT  = 500.0


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

def _get_symbol_info(symbol: str) -> dict:
    url  = f"{BINANCE_TRADE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    for s in resp.json().get("symbols", []):
        if s["symbol"] == symbol.upper():
            filters = {f["filterType"]: f for f in s["filters"]}

            lot    = filters.get("LOT_SIZE", {})
            price  = filters.get("PRICE_FILTER", {})
            notional = filters.get("MIN_NOTIONAL", {})

            info = {
                "qty_step":    float(lot.get("stepSize", "0.001")),
                "min_qty":     float(lot.get("minQty",   "0.001")),
                "max_qty":     float(lot.get("maxQty",   "999999999")),
                "price_tick":  float(price.get("tickSize", "0.0001")),
                "min_notional": float(notional.get("notional", "5")),
            }

            logger.info(
                "[executor] %s filters — stepSize=%s minQty=%s maxQty=%s "
                "tickSize=%s minNotional=%s",
                symbol,
                info["qty_step"], info["min_qty"], info["max_qty"],
                info["price_tick"], info["min_notional"],
            )
            return info

    raise ValueError(f"Symbol {symbol} tidak ditemukan di exchange info")


def _round_step(value: float, step: float) -> float | int:
    if step >= 1.0:
        return int(math.floor(value / step) * step)
    precision = max(0, round(-math.log10(step)))
    return math.floor(value * 10**precision) / 10**precision


def _round_price(value: float, tick: float) -> float:
    from decimal import Decimal, ROUND_DOWN
    tick_dec  = Decimal(str(tick))
    val_dec   = Decimal(str(value))
    rounded   = float(val_dec.quantize(tick_dec, rounding=ROUND_DOWN))
    return rounded


def _get_available_balance() -> float:
    for b in _get("/fapi/v2/balance", {}):
        if b["asset"] == "USDT":
            return float(b["availableBalance"])
    return 0.0


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

def _sltp_worker(
    symbol: str,
    order_id: int,
    side: str,
    sl_side: str,
    tp_side: str,
    qty: float,
    sl_r: float,
    tp_r: float,
    notify_fn,
    sym_info: dict,
) -> None:
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

    deadline     = time.time() + FILL_TIMEOUT_SEC
    filled_qty   = qty
    filled_price = 0.0

    while time.time() < deadline:
        time.sleep(FILL_POLL_INTERVAL)
        try:
            order  = _get("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
            status = order.get("status", "")

            if status == "FILLED":
                raw_filled   = float(order.get("executedQty", qty))
                filled_qty   = _round_step(raw_filled, sym_info["qty_step"])
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
                executed = float(order.get("executedQty", 0))
                if executed > 0:
                    filled_qty   = _round_step(executed, sym_info["qty_step"])
                    filled_price = float(order.get("avgPrice", filled_price))
                    logger.warning(
                        "[executor:bg] Order %d %s tapi executedQty=%s — "
                        "ada posisi terbuka, lanjut pasang SL/TP",
                        order_id, status, filled_qty,
                    )
                    _notify(
                        f"⚠️ <b>{symbol}</b> — Entry order {status} "
                        f"tapi sebagian ter-fill (qty={filled_qty})\n"
                        f"Melanjutkan pasang SL & TP..."
                    )
                    break 
                else:
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

    else:
        logger.warning(
            "[executor:bg] Order %d timeout setelah %ds — cancel",
            order_id, FILL_TIMEOUT_SEC,
        )
        _cancel_order(symbol, order_id)
        _notify(
            f"⏱ <b>{symbol}</b> — Entry order {order_id} tidak ter-fill "
            f"dalam {FILL_TIMEOUT_SEC//60} menit, di-cancel."
        )
    try:
        ticker     = _get("/fapi/v1/premiumIndex", {"symbol": symbol})
        mark_price = float(ticker.get("markPrice", filled_price))
        logger.info("[executor:bg] %s mark price saat pasang SL/TP: %s", symbol, mark_price)
    except Exception as e:
        logger.warning("[executor:bg] Gagal ambil mark price: %s — pakai filled_price", e)
        mark_price = filled_price

    price_tick = sym_info["price_tick"]
    if side == "BUY":
        if sl_r >= mark_price:
            logger.warning(
                "[executor:bg] %s SL %.6f >= mark %.6f — SL sudah terlewat, emergency close",
                symbol, sl_r, mark_price,
            )
            _emergency_close(symbol, side, filled_qty)
            _notify(
                f"🚨 <b>{symbol}</b> — SL {sl_r} sudah terlewat mark price {mark_price:.6f}\n"
                f"Posisi di-close via market order."
            )
            return
    else:
        # SHORT: SL harus di atas mark price
        if sl_r <= mark_price:
            logger.warning(
                "[executor:bg] %s SL %.6f <= mark %.6f — SL sudah terlewat, emergency close",
                symbol, sl_r, mark_price,
            )
            _emergency_close(symbol, side, filled_qty)
            _notify(
                f"🚨 <b>{symbol}</b> — SL {sl_r} sudah terlewat mark price {mark_price:.6f}\n"
                f"Posisi di-close via market order."
            )
            return

    sl_order_id = None
    try:
        sl_resp = _post("/fapi/v1/order", {
            "symbol":        symbol,
            "side":          sl_side,
            "type":          "STOP_MARKET",
            "stopPrice":     sl_r,
            "closePosition": "true",
            "workingType":   "MARK_PRICE",
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

    try:
        tp_resp = _post("/fapi/v1/order", {
            "symbol":        symbol,
            "side":          tp_side,
            "type":          "TAKE_PROFIT_MARKET",
            "stopPrice":     tp_r,
            "closePosition": "true",
            "workingType":   "MARK_PRICE",
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

def execute_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:

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
        _post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

        sym_info  = _get_symbol_info(symbol)
        available = _get_available_balance()

        raw_notional   = available * qty_fraction * leverage
        capped_notional = min(raw_notional, MAX_NOTIONAL_USDT)

        if raw_notional > MAX_NOTIONAL_USDT:
            logger.info(
                "[executor] %s notional capped: %.2f → %.2f USDT (MAX_NOTIONAL)",
                symbol, raw_notional, capped_notional,
            )

        qty = _round_step(capped_notional / entry_price, sym_info["qty_step"])

        entry_r = _round_price(entry_price, sym_info["price_tick"])
        sl_r    = _round_price(stop_loss,   sym_info["price_tick"])
        tp_r    = _round_price(take_profit,  sym_info["price_tick"])

        if qty < sym_info["min_qty"]:
            return {
                "ok": False, "symbol": symbol,
                "reason_fail": (
                    f"Qty {qty} < minQty {sym_info['min_qty']}. "
                    f"Balance={available:.2f} USDT, fraction={qty_fraction:.4f}, "
                    f"notional={capped_notional:.2f} USDT, entry={entry_r}"
                ),
            }

        if qty > sym_info["max_qty"]:
            logger.info(
                "[executor] %s qty %.4f > maxQty %.4f — clamp ke maxQty",
                symbol, qty, sym_info["max_qty"],
            )
            qty = sym_info["max_qty"]
        actual_notional = qty * entry_r
        if actual_notional < sym_info["min_notional"]:
            return {
                "ok": False, "symbol": symbol,
                "reason_fail": (
                    f"Notional {actual_notional:.4f} USDT < minNotional "
                    f"{sym_info['min_notional']} USDT. "
                    f"Qty={qty}, entry={entry_r}. "
                    f"Balance={available:.2f} USDT terlalu kecil untuk token ini."
                ),
            }

        logger.info(
            "[executor] %s order ready — side=%s qty=%s entry=%s "
            "notional=%.2f USDT SL=%s TP=%s lev=%dx",
            symbol, side, qty, entry_r, actual_notional, sl_r, tp_r, leverage,
        )

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

        t = threading.Thread(
            target=_sltp_worker,
            args=(symbol, order_id, side, sl_side, tp_side, qty, sl_r, tp_r, notify_fn, sym_info),
            daemon=True,
            name=f"sltp-{symbol}-{order_id}",
        )
        t.start()
        logger.info("[executor] Background thread started: %s", t.name)

        return {
            "ok":           True,
            "symbol":       symbol,
            "side":         side,
            "order_id":     order_id,
            "sl_order_id":  None,
            "tp_order_id":  None,
            "qty":          qty,
            "entry_price":  entry_r,
            "stop_loss":    sl_r,
            "take_profit":  tp_r,
            "leverage":     leverage,
            "balance_used": round(actual_notional / leverage, 4),
            "notional":     round(actual_notional, 4),
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
