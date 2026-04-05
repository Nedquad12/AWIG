"""
paper_executor.py — Paper trading executor dengan:
  - LIMIT order (bukan market): posisi dicatat sebagai "pending" sampai harga hit entry
  - WebSocket aggTrade memantau apakah harga sudah hit entry
  - Partial TP: saat RR 1.5:1 → close 30% posisi, sisa ikut trailing
  - Re-entry: jika posisi sebelumnya TP di simbol sama → margin dikurangi 35%
  - WTI pct disimpan ke posisi untuk WTI filter di risk_manager
"""

import json
import logging
import os
import time
import uuid
from typing import Optional

import requests

from config import BINANCE_DATA_URL, PAPER_BALANCE_USDT, RISK_PER_TRADE_PCT

logger = logging.getLogger(__name__)

_PROJECT_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_POSITIONS_FILE = os.path.join(_PROJECT_ROOT, "paper_positions.json")
PAPER_HISTORY_FILE   = os.path.join(_PROJECT_ROOT, "paper_history.json")
PAPER_BALANCE_FILE   = os.path.join(_PROJECT_ROOT, "paper_balance.json")

TAKER_FEE           = 0.001
ENTRY_CHECK_CANDLES = 12

# Partial TP config
PARTIAL_TP_RR       = 1.5    # RR threshold untuk partial TP
PARTIAL_TP_PCT      = 0.30   # tutup 30% posisi saat partial TP

# Re-entry penalty
REENTRY_MARGIN_CUT  = 0.35   # kurangi 35% margin jika re-entry setelah TP


# ── File I/O ─────────────────────────────────────────────────────────────────

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
                "available":  round(available, 4),
                "initial":    PAPER_BALANCE_USDT,
                "updated_at": int(time.time()),
            }, f, indent=2)
    except Exception as e:
        logger.error("[paper] Gagal save balance: %s", e)


def get_available_balance() -> float:
    positions   = _load_positions()
    open_pos    = [p for p in positions if p.get("status") in ("open", "pending")]
    used_margin = sum(float(p.get("margin_used", 0)) for p in open_pos)
    available   = PAPER_BALANCE_USDT - used_margin
    return round(max(available, 0), 4)


def has_paper_position(symbol: str) -> bool:
    positions = _load_positions()
    return any(
        p["symbol"] == symbol.upper() and p.get("status") in ("open", "pending")
        for p in positions
    )


# ── Re-entry check ───────────────────────────────────────────────────────────

def _get_reentry_multiplier(symbol: str) -> float:
    """
    Cek apakah posisi terakhir di simbol ini TP.
    Jika iya → return 1 - REENTRY_MARGIN_CUT (margin dikurangi 35%).
    Jika tidak → return 1.0 (normal).
    """
    history = _load_history()
    sym_history = [
        h for h in history
        if h.get("symbol") == symbol.upper()
    ]
    if not sym_history:
        return 1.0
    last = sorted(sym_history, key=lambda x: x.get("closed_at", 0))[-1]
    if last.get("status") == "TP":
        logger.info(
            "[paper] %s re-entry setelah TP — margin dikurangi %.0f%%",
            symbol, REENTRY_MARGIN_CUT * 100,
        )
        return 1.0 - REENTRY_MARGIN_CUT
    return 1.0


# ── Entry reachable check ────────────────────────────────────────────────────

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
            lows    = [float(c[3]) for c in candles]
            closest = min(lows)
            for i, low in enumerate(reversed(lows)):
                if low <= entry_price:
                    reachable        = True
                    touch_candle_idx = i
                    break
            closest_pct = (entry_price - closest) / entry_price * 100
        else:
            highs   = [float(c[2]) for c in candles]
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
        }


# ── Partial TP helper ─────────────────────────────────────────────────────────

def check_partial_tp(pos: dict, current_price: float) -> bool:
    """
    Cek apakah partial TP sudah harus dieksekusi.
    Kondisi: harga mencapai RR >= PARTIAL_TP_RR dan belum partial TP.
    """
    if pos.get("partial_tp_done"):
        return False

    entry = float(pos.get("entry_price", 0))
    sl    = float(pos.get("stop_loss", 0))
    tp    = float(pos.get("take_profit", 0))
    side  = pos.get("side", "BUY")

    if entry <= 0 or sl <= 0:
        return False

    risk   = abs(entry - sl)
    if risk <= 0:
        return False

    # Harga partial TP = entry + 1.5 × risk (LONG) atau entry - 1.5 × risk (SHORT)
    if side == "BUY":
        partial_tp_price = entry + PARTIAL_TP_RR * risk
        return current_price >= partial_tp_price
    else:
        partial_tp_price = entry - PARTIAL_TP_RR * risk
        return current_price <= partial_tp_price


def execute_partial_tp(pos: dict, current_price: float, notify_fn=None) -> dict:
    """
    Close 30% posisi saat RR 1.5:1 tercapai.
    Update posisi: qty dan notional berkurang 30%, partial_tp_done=True.
    Simpan partial close ke history.
    Return updated position dict.
    """
    def _notify(msg: str):
        if notify_fn:
            try:
                notify_fn(msg)
            except Exception:
                pass

    partial_qty      = round(float(pos.get("qty", 0)) * PARTIAL_TP_PCT, 6)
    partial_notional = float(pos.get("notional", 0)) * PARTIAL_TP_PCT
    entry            = float(pos.get("entry_price", 0))
    side             = pos.get("side", "BUY")

    if side == "BUY":
        partial_pnl = (current_price - entry) / entry * partial_notional
    else:
        partial_pnl = (entry - current_price) / entry * partial_notional
    partial_pnl -= partial_notional * TAKER_FEE * 2
    partial_pnl  = round(partial_pnl, 4)

    # Update posisi: kurangi qty dan notional
    new_qty      = round(float(pos.get("qty", 0))      * (1 - PARTIAL_TP_PCT), 6)
    new_notional = round(float(pos.get("notional", 0)) * (1 - PARTIAL_TP_PCT), 4)
    new_margin   = round(new_notional / float(pos.get("leverage", 1)), 4)

    pos_updated = dict(pos)
    pos_updated["qty"]           = new_qty
    pos_updated["notional"]      = new_notional
    pos_updated["margin_used"]   = new_margin
    pos_updated["partial_tp_done"]  = True
    pos_updated["partial_tp_price"] = current_price
    pos_updated["partial_tp_pnl"]   = partial_pnl

    # Simpan partial close ke history sebagai event terpisah
    partial_record = dict(pos)
    partial_record["status"]        = "PARTIAL_TP"
    partial_record["close_reason"]  = f"Partial TP {PARTIAL_TP_PCT*100:.0f}% @ RR {PARTIAL_TP_RR}"
    partial_record["close_price"]   = current_price
    partial_record["pnl"]           = partial_pnl
    partial_record["closed_at"]     = time.time()
    partial_record["qty_closed"]    = partial_qty
    partial_record["notional_closed"] = round(partial_notional, 4)
    _append_history(partial_record)

    # Update posisi di file
    positions = _load_positions()
    for i, p in enumerate(positions):
        if p.get("paper_id") == pos.get("paper_id"):
            positions[i] = pos_updated
            break
    _save_positions(positions)

    pnl_str = f"+{partial_pnl:.2f}" if partial_pnl >= 0 else f"{partial_pnl:.2f}"
    _notify(
        f"🎯 <b>Partial TP {PARTIAL_TP_PCT*100:.0f}% — {pos.get('symbol')}</b>\n"
        f"  Price  : <code>{current_price:.6f}</code>  (RR {PARTIAL_TP_RR}:1)\n"
        f"  PnL    : <b>{pnl_str} USDT</b>\n"
        f"  Sisa   : {(1-PARTIAL_TP_PCT)*100:.0f}% posisi masih aktif\n"
        f"  SL geser ke entry (breakeven) ✅"
    )
    logger.info(
        "[paper] Partial TP %s @ %.6f pnl=%.4f qty_remaining=%.6f",
        pos.get("symbol"), current_price, partial_pnl, new_qty,
    )
    return pos_updated


# ── Main executor ─────────────────────────────────────────────────────────────

def execute_paper_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """
    Buat paper order sebagai LIMIT (status="pending").
    Monitor di monitor.py akan pantau via WS apakah entry price tercapai.
    Saat tercapai → status diubah ke "open".
    """

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
    wti_pct      = float(ai_result.get("wti_pct", 0.0))

    side = "BUY" if action == "BUYING" else "SELL"

    # Cek current price — kalau entry sudah kelewat jauh, skip
    entry_check = _check_entry_reachable(symbol, entry_price, side)

    # Re-entry penalty
    reentry_mult = _get_reentry_multiplier(symbol)
    if reentry_mult < 1.0:
        qty_fraction = round(qty_fraction * reentry_mult, 6)
        qty_fraction = max(0.001, qty_fraction)

    available   = get_available_balance()
    notional    = available * qty_fraction * leverage
    qty         = round(notional / entry_price, 6)
    margin_used = round(notional / leverage, 4)
    fee_open    = round(notional * TAKER_FEE, 4)

    if margin_used > available:
        _notify(
            f"⚠️ <b>{symbol}</b> — Skip: margin <code>{margin_used:.2f} USDT</code> "
            f"melebihi saldo <code>{available:.2f} USDT</code>"
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

    # Hitung harga partial TP untuk info
    risk = abs(entry_price - stop_loss)
    if side == "BUY":
        partial_tp_price = round(entry_price + PARTIAL_TP_RR * risk, 6)
    else:
        partial_tp_price = round(entry_price - PARTIAL_TP_RR * risk, 6)

    position = {
        "paper_id":          paper_id,
        "symbol":            symbol,
        "side":              side,
        "entry_price":       entry_price,
        "stop_loss":         stop_loss,
        "take_profit":       take_profit,
        "leverage":          leverage,
        "qty":               qty,
        "notional":          round(notional, 4),
        "margin_used":       margin_used,
        "qty_fraction":      round(qty_fraction, 6),
        "fee_open":          fee_open,
        "pnl_if_tp":         pnl_tp,
        "pnl_if_sl":         pnl_sl,
        "rr":                rr,
        "entry_check":       entry_check,
        "opened_at":         opened_at,
        "status":            "pending",       # ← LIMIT: tunggu entry hit via WS
        "partial_tp_price":  partial_tp_price,
        "partial_tp_done":   False,
        "wti_pct":           wti_pct,          # disimpan untuk WTI filter
        "reentry_mult":      reentry_mult,
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)

    logger.info(
        "[paper] 📝 LIMIT %s %s entry=%.6f SL=%.6f TP=%.6f lev=%dx notional=%.2f wti=%.1f%%",
        side, symbol, entry_price, stop_loss, take_profit, leverage, notional, wti_pct,
    )

    side_emoji    = "🟢" if side == "BUY" else "🔴"
    pnl_tp_str    = f"+{pnl_tp:.2f}" if pnl_tp > 0 else f"{pnl_tp:.2f}"
    reentry_note  = f"\n  ⚠️ Re-entry: margin dikurangi {REENTRY_MARGIN_CUT*100:.0f}%" if reentry_mult < 1.0 else ""
    entry_status  = ""
    if entry_check["candles_checked"] > 0:
        if entry_check["reachable"]:
            entry_status = f"\n  Entry 5m : ✅ pernah menyentuh"
        else:
            entry_status = f"\n  Entry 5m : ⏳ belum hit, monitor WS aktif (gap {entry_check['closest_pct']:.2f}%)"

    sisa_modal = get_available_balance()

    _notify(
        f"📋 <b>PAPER LIMIT ORDER — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  {side_emoji} <b>{side}</b>  ×{leverage}  |  ID: <code>{paper_id}</code>\n"
        f"  Entry    : <code>{entry_price}</code>  <i>(menunggu hit)</i>\n"
        f"  SL       : <code>{stop_loss}</code>\n"
        f"  TP Full  : <code>{take_profit}</code>  (RR {rr})\n"
        f"  TP 30%   : <code>{partial_tp_price}</code>  (RR {PARTIAL_TP_RR}:1)\n"
        f"  Notional : <code>{notional:.2f} USDT</code>  margin: <code>{margin_used} USDT</code>\n"
        f"  WTI      : <code>{wti_pct:.1f}%</code>\n"
        f"  Est TP   : <b>{pnl_tp_str} USDT</b>  |  Est SL: <b>{pnl_sl:.2f} USDT</b>"
        f"{entry_status}{reentry_note}\n"
        f"  💰 Saldo tersedia: <b>{sisa_modal:.2f} USDT</b>"
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
        "wti_pct":      wti_pct,
        "note":         "PAPER LIMIT — entry dipantau via WebSocket",
    }


def activate_pending_position(paper_id: str, fill_price: float, notify_fn=None) -> bool:
    """
    Ubah status posisi dari "pending" → "open" saat harga WS hit entry.
    Dipanggil dari monitor.py saat aggTrade price menyentuh entry.
    Return True jika berhasil.
    """
    def _notify(msg: str):
        if notify_fn:
            try:
                notify_fn(msg)
            except Exception:
                pass

    positions = _load_positions()
    for i, pos in enumerate(positions):
        if pos.get("paper_id") == paper_id and pos.get("status") == "pending":
            positions[i]["status"]    = "open"
            positions[i]["fill_price"] = fill_price
            positions[i]["filled_at"]  = int(time.time())
            _save_positions(positions)

            sym   = pos.get("symbol", "")
            side  = pos.get("side", "BUY")
            emoji = "🟢" if side == "BUY" else "🔴"
            _notify(
                f"{emoji} <b>LIMIT FILLED — {sym}</b>\n"
                f"  ID     : <code>{paper_id}</code>\n"
                f"  Fill   : <code>{fill_price:.6f}</code>\n"
                f"  SL     : <code>{pos['stop_loss']}</code>\n"
                f"  TP     : <code>{pos['take_profit']}</code>\n"
                f"  TP 30% : <code>{pos.get('partial_tp_price', '?')}</code>\n"
                f"  Posisi aktif ✅"
            )
            logger.info("[paper] LIMIT FILLED %s %s @ %.6f", paper_id, sym, fill_price)
            return True

    logger.warning("[paper] activate_pending: paper_id=%s tidak ditemukan", paper_id)
    return False


def cancel_pending_position(paper_id: str, reason: str = "expired", notify_fn=None) -> bool:
    """
    Batalkan posisi pending (misal: terlalu lama tidak ter-fill).
    Return True jika berhasil.
    """
    def _notify(msg: str):
        if notify_fn:
            try:
                notify_fn(msg)
            except Exception:
                pass

    positions = _load_positions()
    for i, pos in enumerate(positions):
        if pos.get("paper_id") == paper_id and pos.get("status") == "pending":
            positions[i]["status"]      = "cancelled"
            positions[i]["cancel_reason"] = reason
            positions[i]["cancelled_at"]  = int(time.time())
            _save_positions(positions)
            sym = pos.get("symbol", "")
            _notify(
                f"❌ <b>LIMIT CANCELLED — {sym}</b>\n"
                f"  ID     : <code>{paper_id}</code>\n"
                f"  Reason : {reason}"
            )
            logger.info("[paper] LIMIT CANCELLED %s %s reason=%s", paper_id, sym, reason)
            return True

    return False
