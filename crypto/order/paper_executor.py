import json
import logging
import os
import time
import uuid

import requests

from config import BINANCE_DATA_URL, PAPER_BALANCE_USDT, RISK_PER_TRADE_PCT

logger = logging.getLogger(__name__)

_PROJECT_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_POSITIONS_FILE = os.path.join(_PROJECT_ROOT, "paper_positions.json")
PAPER_HISTORY_FILE   = os.path.join(_PROJECT_ROOT, "paper_history.json")
PAPER_BALANCE_FILE   = os.path.join(_PROJECT_ROOT, "paper_balance.json")

TAKER_FEE = 0.001 

ENTRY_CHECK_CANDLES = 12 

def _load_positions() -> list:
    try:
        if os.path.exists(PAPER_POSITIONS_FILE):
            with open(PAPER_POSITIONS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error("[paper] Gagal load positions: %s", e)
    return []


def _save_positions(positions: list) -> None:
    try:
        with open(PAPER_POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except Exception as e:
        logger.error("[paper] Gagal save positions: %s", e)


def _load_history() -> list:
    try:
        if os.path.exists(PAPER_HISTORY_FILE):
            with open(PAPER_HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error("[paper] Gagal load history: %s", e)
    return []


def _append_history(record: dict) -> None:
    history = _load_history()
    history.append(record)
    try:
        with open(PAPER_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error("[paper] Gagal save history: %s", e)

def _load_balance() -> float:
    """Load saldo tersedia dari file. Default = PAPER_BALANCE_USDT."""
    try:
        if os.path.exists(PAPER_BALANCE_FILE):
            with open(PAPER_BALANCE_FILE) as f:
                data = json.load(f)
            return float(data.get("available", PAPER_BALANCE_USDT))
    except Exception:
        pass
    return PAPER_BALANCE_USDT


def _save_balance(available: float) -> None:
    try:
        with open(PAPER_BALANCE_FILE, "w") as f:
            json.dump({
                "available":   round(available, 4),
                "initial":     PAPER_BALANCE_USDT,
                "updated_at":  int(time.time()),
            }, f, indent=2)
    except Exception as e:
        logger.error("[paper] Gagal save balance: %s", e)


def get_available_balance() -> float:
    """
    Saldo tersedia = PAPER_BALANCE_USDT dikurangi semua margin posisi aktif.
    Hitung ulang dari posisi aktif agar selalu akurat.
    """
    positions  = _load_positions()
    open_pos   = [p for p in positions if p.get("status") == "open"]
    used_margin = sum(float(p.get("margin_used", 0)) for p in open_pos)
    available  = PAPER_BALANCE_USDT - used_margin
    return round(max(available, 0), 4)

def has_paper_position(symbol: str) -> bool:
    positions = _load_positions()
    return any(p["symbol"] == symbol.upper() and p.get("status") == "open" for p in positions)

def _check_entry_reachable(symbol: str, entry_price: float, side: str) -> dict:
    try:
        resp = requests.get(
            f"{BINANCE_DATA_URL}/fapi/v1/klines",
            params={"symbol": symbol, "interval": "5m", "limit": ENTRY_CHECK_CANDLES + 1},
            timeout=10,
        )
        resp.raise_for_status()
        candles = resp.json()

        if len(candles) > ENTRY_CHECK_CANDLES:
            candles = candles[:-1]

        reachable        = False
        touch_candle_idx = None

        if side == "BUY":
            lows = [float(c[3]) for c in candles]
            closest = min(lows)
            for i, low in enumerate(reversed(lows)):
                if low <= entry_price:
                    reachable        = True
                    touch_candle_idx = i
                    break
            closest_pct = (entry_price - closest) / entry_price * 100
        else:
            highs = [float(c[2]) for c in candles]
            closest = max(highs)
            for i, high in enumerate(reversed(highs)):
                if high >= entry_price:
                    reachable        = True
                    touch_candle_idx = i
                    break
            closest_pct = (closest - entry_price) / entry_price * 100

        return {
            "reachable":        reachable,
            "closest_price":    round(closest, 8),
            "closest_pct":      round(closest_pct, 3),
            "candles_checked":  len(candles),
            "touch_candle_idx": touch_candle_idx,
        }

    except Exception as e:
        logger.warning("[paper] Gagal cek entry 5m %s: %s — anggap reachable", symbol, e)
        return {
            "reachable":        True,
            "closest_price":    entry_price,
            "closest_pct":      0.0,
            "candles_checked":  0,
            "touch_candle_idx": None,
            "error":            str(e),
        }

def execute_paper_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:

    def _notify(msg: str):
        if notify_fn:
            try:
                notify_fn(msg)
            except Exception as e:
                logger.debug("[paper] notify error: %s", e)

    symbol       = pred["symbol"].upper()
    action       = ai_result["action"]
    entry_price  = float(ai_result["entry_price"])
    stop_loss    = float(ai_result["stop_loss"])
    take_profit  = float(ai_result["take_profit"])
    leverage     = int(ai_result["leverage"])
    qty_fraction = float(ai_result.get("qty_fraction", RISK_PER_TRADE_PCT / 100))
    qty_fraction = max(0.001, min(qty_fraction, 1.0))

    side = "BUY" if action == "BUYING" else "SELL"
    entry_check = _check_entry_reachable(symbol, entry_price, side)
    entry_status_line = ""

    if entry_check["candles_checked"] > 0:
        window_min = entry_check["candles_checked"] * 5
        if entry_check["reachable"]:
            idx = entry_check["touch_candle_idx"]
            mins_ago = (idx + 1) * 5 if idx is not None else "?"
            entry_status_line = f"  Entry 5m    : ✅ Pernah menyentuh ~{mins_ago} menit lalu"
            logger.info("[paper] %s entry %.6f REACHED dalam %dm", symbol, entry_price, window_min)
        else:
            pct     = entry_check["closest_pct"]
            closest = entry_check["closest_price"]
            entry_status_line = (
                f"  Entry 5m    : ⚠️ Belum menyentuh dalam {window_min}m "
                f"(closest {closest:.6f}, jarak {pct:.2f}%)"
            )
            logger.warning(
                "[paper] %s entry %.6f BELUM tercapai dalam %dm — closest=%.6f (%.3f%%)",
                symbol, entry_price, window_min, closest, pct,
            )
    else:
        entry_status_line = "  Entry 5m    : ⚪ Tidak dapat fetch candle"
    available   = get_available_balance()
    notional    = available * qty_fraction * leverage
    qty         = round(notional / entry_price, 6)
    margin_used = round(notional / leverage, 4)
    fee_open    = round(notional * TAKER_FEE, 4)
    if margin_used > available:
        _notify(
            f"⚠️ <b>{symbol}</b> — Skip: margin <code>{margin_used:.2f} USDT</code> "
            f"melebihi saldo tersedia <code>{available:.2f} USDT</code>"
        )
        return {"ok": False, "reason_fail": f"Margin {margin_used:.2f} > available {available:.2f}"}

    if margin_used < 1.0:
        _notify(f"⚠️ <b>{symbol}</b> — Skip: saldo tidak cukup (<code>{available:.2f} USDT</code>)")
        return {"ok": False, "reason_fail": f"Saldo tidak cukup: {available:.2f} USDT"}

    if side == "BUY":
        pnl_tp = round((take_profit - entry_price) / entry_price * notional - fee_open * 2, 4)
        pnl_sl = round((stop_loss  - entry_price) / entry_price * notional - fee_open * 2, 4)
    else:
        pnl_tp = round((entry_price - take_profit) / entry_price * notional - fee_open * 2, 4)
        pnl_sl = round((entry_price - stop_loss)   / entry_price * notional - fee_open * 2, 4)

    rr = round(abs(take_profit - entry_price) / abs(stop_loss - entry_price), 2)

    paper_id  = str(uuid.uuid4())[:8].upper()
    opened_at = int(time.time())

    position = {
        "paper_id":     paper_id,
        "symbol":       symbol,
        "side":         side,
        "entry_price":  entry_price,
        "stop_loss":    stop_loss,
        "take_profit":  take_profit,
        "leverage":     leverage,
        "qty":          qty,
        "notional":     round(notional, 4),
        "margin_used":  margin_used,
        "qty_fraction": round(qty_fraction, 6),
        "fee_open":     fee_open,
        "pnl_if_tp":    pnl_tp,
        "pnl_if_sl":    pnl_sl,
        "rr":           rr,
        "entry_check":  entry_check,
        "opened_at":    opened_at,
        "status":       "open",
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)

    logger.info(
        "[paper] 📝 %s %s entry=%.6f SL=%.6f TP=%.6f lev=%dx notional=%.2f",
        side, symbol, entry_price, stop_loss, take_profit, leverage, notional,
    )

    side_emoji = "🟢" if side == "BUY" else "🔴"
    pnl_tp_str = f"+{pnl_tp:.2f}" if pnl_tp > 0 else f"{pnl_tp:.2f}"
    sisa_modal = get_available_balance()

    _notify(
        f"📝 <b>PAPER TRADE — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  {side_emoji} <b>{side}</b>  ×{leverage}  |  ID: <code>{paper_id}</code>\n"
        f"  Entry  : <code>{entry_price}</code>\n"
        f"  SL     : <code>{stop_loss}</code>   TP: <code>{take_profit}</code>  (RR {rr})\n"
        f"  Notional: <code>{notional:.2f} USDT</code>  margin: <code>{margin_used} USDT</code>\n"
        f"  🎯 TP → <b>{pnl_tp_str} USDT</b>  |  🛑 SL → <b>{pnl_sl:.2f} USDT</b>\n"
        f"{entry_status_line}\n"
        f"  💰 Saldo tersedia: <b>{sisa_modal:.2f} USDT</b> (dari {PAPER_BALANCE_USDT:,.0f})"
    )

    return {
        "ok":           True,
        "paper":        True,
        "symbol":       symbol,
        "side":         side,
        "order_id":     paper_id,
        "qty":          qty,
        "entry_price":  entry_price,
        "stop_loss":    stop_loss,
        "take_profit":  take_profit,
        "leverage":     leverage,
        "balance_used": margin_used,
        "notional":     round(notional, 4),
        "qty_fraction": round(qty_fraction, 6),
        "note":         "PAPER TRADE — tidak ada order nyata di Binance",
    }
