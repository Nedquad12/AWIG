"""
order/live_executor.py — Modul order khusus live Binance Futures.

Arsitektur:
  - Selalu kirim ke Binance (tidak ada paper mode)
  - Limit order  : kirim bracket (entry LIMIT + SL + TP) ke Binance
                   simpan ke positions.json dengan status=pending
                   monitor.py polling Binance untuk cek fill
  - Market order : kirim MARKET ke Binance, SL/TP langsung aktif
                   simpan ke positions.json dengan status=open
  - positions.json = sumber kebenaran tunggal → monitor baca dari sini

Proteksi berlapis:
  1. Volume Analyzer  → close paksa (via monitor → Binance market)
  2. Trailing stop    → SL utama (via monitor → Binance market)
  3. Breakeven SL     → amend SL Binance ke entry saat breakeven hit
  4. SL/TP bracket    → Binance conditional order, fallback jika sistem down
"""

import json
import logging
import os
import time
import uuid
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
_ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POSITIONS_FILE = os.path.join(_ROOT, "positions.json")
HISTORY_FILE   = os.path.join(_ROOT, "positions_history.json")

# ── Biaya & konstanta ─────────────────────────────────────────────────────────
TAKER_FEE      = 0.0004
PARTIAL_TP_RR  = 1.5
PARTIAL_TP_PCT = 0.30
REENTRY_MARGIN_CUT = 0.35

from config import RISK_PER_TRADE_PCT


# ─────────────────────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_positions() -> list:
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE) as f:
                return json.load(f)
    except Exception as e:
        logger.error("[live] Gagal load positions: %s", e)
    return []


def _save_positions(positions: list) -> None:
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except Exception as e:
        logger.error("[live] Gagal save positions: %s", e)


def _load_history() -> list:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception as e:
        logger.error("[live] Gagal load history: %s", e)
    return []


def _append_history(record: dict) -> None:
    hist = _load_history()
    hist.append(record)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(hist, f, indent=2)
    except Exception as e:
        logger.error("[live] Gagal save history: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Balance & helper
# ─────────────────────────────────────────────────────────────────────────────

def get_available_balance() -> float:
    """Ambil available balance USDT dari Binance."""
    from order.executor import get_available_balance as _live
    return _live()


def has_position(symbol: str) -> bool:
    """Cek apakah sudah ada posisi open/pending untuk symbol."""
    return any(
        p["symbol"] == symbol.upper() and p.get("status") in ("open", "pending")
        for p in _load_positions()
    )


def _clamp_leverage(symbol: str, leverage: int) -> int:
    from order.executor import get_max_leverage
    max_lev = get_max_leverage(symbol)
    if leverage > max_lev:
        logger.info("[live] %s leverage %dx > max %dx → clamp", symbol, leverage, max_lev)
        return max_lev
    return leverage


def _get_reentry_multiplier(symbol: str) -> float:
    """Kurangi margin 35% jika re-entry setelah TP."""
    hist = _load_history()
    sym_hist = [h for h in hist if h.get("symbol") == symbol.upper()]
    if not sym_hist:
        return 1.0
    last = sorted(sym_hist, key=lambda x: x.get("closed_at", 0))[-1]
    if last.get("status") == "TP":
        logger.info("[live] %s re-entry setelah TP → margin -35%%", symbol)
        return 1.0 - REENTRY_MARGIN_CUT
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi publik: count_session_filled (dipanggil monitor)
# ─────────────────────────────────────────────────────────────────────────────

def count_session_filled(session_id: str) -> int:
    if not session_id:
        return 0
    return sum(
        1 for p in _load_positions()
        if p.get("session_id") == session_id and p.get("status") == "open"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LIMIT order — bracket ke Binance
# ─────────────────────────────────────────────────────────────────────────────

def execute_limit_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """
    Kirim LIMIT bracket order ke Binance dan simpan ke positions.json.

    Flow:
      1. Validasi arah (SL/TP vs entry)
      2. Hitung qty dari balance & leverage
      3. Kirim batch: LIMIT entry + STOP_MARKET SL + TAKE_PROFIT_MARKET TP
      4. Simpan ke positions.json status=pending
      5. monitor.py akan poll Binance tiap 5 detik untuk cek fill
    """
    def _n(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    from order.executor import (
        get_symbol_info, set_leverage, place_bracket_order,
        round_step, round_price, MAX_NOTIONAL_USDT,
    )

    symbol       = pred["symbol"].upper()
    action       = ai_result["action"]
    entry_price  = float(ai_result["entry_price"])
    stop_loss    = float(ai_result["stop_loss"])
    take_profit  = float(ai_result["take_profit"])
    leverage     = int(ai_result["leverage"])
    qty_fraction = float(ai_result.get("qty_fraction", RISK_PER_TRADE_PCT / 100))
    qty_fraction = max(0.001, min(qty_fraction, 1.0))
    wti_pct      = float(ai_result.get("wti_pct", 0.0))
    session_id   = ai_result.get("session_id", "")

    side = "BUY" if action == "BUYING" else "SELL"

    # Validasi arah
    if side == "BUY":
        if stop_loss >= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"SL {stop_loss} >= entry {entry_price}"}
        if take_profit <= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"TP {take_profit} <= entry {entry_price}"}
    else:
        if stop_loss <= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"SL {stop_loss} <= entry {entry_price}"}
        if take_profit >= entry_price:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"TP {take_profit} >= entry {entry_price}"}

    leverage = _clamp_leverage(symbol, leverage)

    available = get_available_balance()
    if available <= 0:
        return {"ok": False, "symbol": symbol,
                "reason_fail": f"Balance tidak cukup: {available:.2f}"}

    reentry_mult = _get_reentry_multiplier(symbol)
    if reentry_mult < 1.0:
        qty_fraction = max(0.001, round(qty_fraction * reentry_mult, 6))

    try:
        sym_info = get_symbol_info(symbol)
    except Exception as e:
        return {"ok": False, "symbol": symbol, "reason_fail": f"symbol_info error: {e}"}

    entry_r = round_price(entry_price, sym_info["price_tick"])
    sl_r    = round_price(stop_loss,   sym_info["price_tick"])
    tp_r    = round_price(take_profit, sym_info["price_tick"])

    raw_notional = available * qty_fraction * leverage
    capped       = min(raw_notional, MAX_NOTIONAL_USDT)
    qty = round_step(capped / entry_r, sym_info["qty_step"])
    qty = max(sym_info["min_qty"], min(qty, sym_info["max_qty"]))

    actual_notional = qty * entry_r
    if actual_notional < sym_info["min_notional"]:
        return {"ok": False, "symbol": symbol,
                "reason_fail": (f"Notional {actual_notional:.4f} < min "
                                f"{sym_info['min_notional']} USDT. Balance terlalu kecil.")}

    margin_used = round(actual_notional / leverage, 4)
    risk        = abs(entry_r - sl_r)
    partial_tp  = round(
        (entry_r + PARTIAL_TP_RR * risk) if side == "BUY"
        else (entry_r - PARTIAL_TP_RR * risk), 8
    )
    rr = round(abs(tp_r - entry_r) / risk, 2) if risk > 0 else 0

    # Kirim bracket ke Binance
    try:
        leverage = set_leverage(symbol, leverage)
        bracket  = place_bracket_order(symbol, side, qty, entry_r, sl_r, tp_r)
    except Exception as e:
        return {"ok": False, "symbol": symbol, "reason_fail": f"Bracket order gagal: {e}"}

    order_id = str(uuid.uuid4())[:12]
    now_ts   = int(time.time())

    position = {
        "order_id":         order_id,
        "status":           "pending",
        "symbol":           symbol,
        "side":             side,
        "entry_price":      entry_r,
        "stop_loss":        sl_r,
        "sl_initial":       sl_r,
        "take_profit":      tp_r,
        "partial_tp_price": partial_tp,
        "leverage":         leverage,
        "qty":              qty,
        "notional":         round(actual_notional, 4),
        "margin_used":      margin_used,
        "qty_fraction":     round(qty_fraction, 6),
        "wti_pct":          wti_pct,
        "session_id":       session_id,
        "opened_at":        now_ts,
        "filled_at":        0,
        "breakeven_hit":    False,
        "partial_tp_done":  False,
        "binance_order_id": bracket["entry_order_id"],
        "sl_order_id":      bracket["sl_order_id"],
        "tp_order_id":      bracket["tp_order_id"],
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)

    pnl_tp = round(
        (tp_r - entry_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2
        if side == "BUY" else
        (entry_r - tp_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2, 4
    )
    pnl_sl = round(
        (sl_r - entry_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2
        if side == "BUY" else
        (entry_r - sl_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2, 4
    )
    pnl_tp_str = f"+{pnl_tp:.2f}" if pnl_tp > 0 else f"{pnl_tp:.2f}"
    side_emoji = "🟢" if side == "BUY" else "🔴"

    _n(
        f"📋 <b>LIVE LIMIT ORDER — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  {side_emoji} <b>{side}</b>  ×{leverage}  |  ID: <code>{order_id}</code>\n"
        f"  Entry    : <code>{entry_r}</code>  <i>(menunggu fill)</i>\n"
        f"  SL       : <code>{sl_r}</code>  ✅ aktif di Binance\n"
        f"  TP Full  : <code>{tp_r}</code>  (RR {rr}:1)  ✅ aktif\n"
        f"  TP 30%   : <code>{partial_tp:.8f}</code>  (RR {PARTIAL_TP_RR}:1)\n"
        f"  Notional : <code>{actual_notional:.2f} USDT</code>  margin: <code>{margin_used} USDT</code>\n"
        f"  WTI      : <code>{wti_pct:.1f}%</code>\n"
        f"  Est TP   : <b>{pnl_tp_str} USDT</b>  |  Est SL: <b>{pnl_sl:.2f} USDT</b>"
        + (f"\n  ⚠️ Re-entry: margin -{REENTRY_MARGIN_CUT*100:.0f}%" if reentry_mult < 1.0 else "")
    )

    logger.info("[live] Limit order %s %s entry=%.6f sl=%.6f tp=%.6f binance_id=%s",
                symbol, side, entry_r, sl_r, tp_r, bracket["entry_order_id"])

    return {
        "ok":           True,
        "order_type":   "LIMIT",
        "symbol":       symbol,
        "side":         side,
        "order_id":     order_id,
        "qty":          qty,
        "entry_price":  entry_r,
        "stop_loss":    sl_r,
        "take_profit":  tp_r,
        "leverage":     leverage,
        "balance_used": margin_used,
        "notional":     round(actual_notional, 4),
        "qty_fraction": round(qty_fraction, 6),
        "wti_pct":      wti_pct,
        "note":         "LIVE BRACKET — SL/TP aktif di Binance sejak awal ✅",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MARKET order — fill langsung
# ─────────────────────────────────────────────────────────────────────────────

def execute_market_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """
    Kirim MARKET order ke Binance, pasang SL/TP, simpan status=open.
    Dipakai untuk special coin (fill langsung, tidak tunggu pending).
    """
    def _n(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    from order.executor import (
        get_symbol_info, set_leverage, place_market_order,
        place_stop_market, place_take_profit_market,
        cancel_algo_order, close_position_market,
        get_mark_price, round_step, round_price, MAX_NOTIONAL_USDT,
    )

    symbol       = pred["symbol"].upper()
    action       = ai_result["action"]
    stop_loss    = float(ai_result["stop_loss"])
    take_profit  = float(ai_result["take_profit"])
    leverage     = int(ai_result["leverage"])
    qty_fraction = float(ai_result.get("qty_fraction", RISK_PER_TRADE_PCT / 100))
    qty_fraction = max(0.001, min(qty_fraction, 1.0))
    wti_pct      = float(ai_result.get("wti_pct", 0.0))
    session_id   = ai_result.get("session_id", "")

    side = "BUY" if action == "BUYING" else "SELL"
    leverage = _clamp_leverage(symbol, leverage)

    mark_price = get_mark_price(symbol)
    if not mark_price:
        return {"ok": False, "symbol": symbol, "reason_fail": "Gagal ambil mark price"}

    available = get_available_balance()
    if available <= 0:
        return {"ok": False, "symbol": symbol,
                "reason_fail": f"Balance tidak cukup: {available:.2f}"}

    reentry_mult = _get_reentry_multiplier(symbol)
    if reentry_mult < 1.0:
        qty_fraction = max(0.001, round(qty_fraction * reentry_mult, 6))

    try:
        sym_info = get_symbol_info(symbol)
    except Exception as e:
        return {"ok": False, "symbol": symbol, "reason_fail": f"symbol_info error: {e}"}

    raw_notional = available * qty_fraction * leverage
    capped       = min(raw_notional, MAX_NOTIONAL_USDT)
    qty = round_step(capped / mark_price, sym_info["qty_step"])
    qty = max(sym_info["min_qty"], min(qty, sym_info["max_qty"]))

    sl_r  = round_price(stop_loss,   sym_info["price_tick"])
    tp_r  = round_price(take_profit, sym_info["price_tick"])
    risk  = abs(mark_price - sl_r)
    partial_tp = round(
        (mark_price + PARTIAL_TP_RR * risk) if side == "BUY"
        else (mark_price - PARTIAL_TP_RR * risk), 8
    )

    actual_notional = qty * mark_price
    margin_used     = round(actual_notional / leverage, 4)

    # Kirim MARKET ke Binance
    try:
        leverage  = set_leverage(symbol, leverage)
        mkt_resp  = place_market_order(symbol, side, qty)
        fill_price = float(mkt_resp.get("avgPrice", mark_price)) or mark_price
        binance_oid = mkt_resp.get("orderId")
    except Exception as e:
        return {"ok": False, "symbol": symbol, "reason_fail": f"Market order gagal: {e}"}

    # Pasang SL
    sl_side = "SELL" if side == "BUY" else "BUY"
    sl_order_id = None
    tp_order_id = None

    try:
        sl_resp     = place_stop_market(symbol, sl_side, sl_r, close_position=True)
        sl_order_id = sl_resp.get("algoId")
    except Exception as e:
        logger.error("[live:market] SL gagal %s: %s", symbol, e)
        try: close_position_market(symbol, side, qty)
        except Exception: pass
        _n(f"🚨 <b>{symbol}</b> — MARKET SL gagal: <code>{e}</code>")
        return {"ok": False, "symbol": symbol, "reason_fail": f"SL gagal: {e}"}

    try:
        tp_resp     = place_take_profit_market(symbol, sl_side, tp_r, close_position=True)
        tp_order_id = tp_resp.get("algoId")
    except Exception as e:
        logger.error("[live:market] TP gagal %s: %s", symbol, e)
        cancel_algo_order(symbol, sl_order_id)
        try: close_position_market(symbol, side, qty)
        except Exception: pass
        _n(f"🚨 <b>{symbol}</b> — MARKET TP gagal: <code>{e}</code>")
        return {"ok": False, "symbol": symbol, "reason_fail": f"TP gagal: {e}"}

    order_id = str(uuid.uuid4())[:12]
    now_ts   = int(time.time())

    position = {
        "order_id":         order_id,
        "status":           "open",
        "symbol":           symbol,
        "side":             side,
        "entry_price":      fill_price,
        "fill_price":       fill_price,
        "stop_loss":        sl_r,
        "sl_initial":       sl_r,
        "take_profit":      tp_r,
        "partial_tp_price": partial_tp,
        "leverage":         leverage,
        "qty":              qty,
        "notional":         round(actual_notional, 4),
        "margin_used":      margin_used,
        "qty_fraction":     round(qty_fraction, 6),
        "wti_pct":          wti_pct,
        "session_id":       session_id,
        "opened_at":        now_ts,
        "filled_at":        now_ts,
        "breakeven_hit":    False,
        "partial_tp_done":  False,
        "special_coin":     True,
        "binance_order_id": binance_oid,
        "sl_order_id":      sl_order_id,
        "tp_order_id":      tp_order_id,
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)

    side_emoji = "🟢" if side == "BUY" else "🔴"
    _n(
        f"{side_emoji} <b>LIVE MARKET ORDER — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  ID      : <code>{order_id}</code>\n"
        f"  Fill    : <code>{fill_price:.6f}</code>\n"
        f"  SL      : <code>{sl_r}</code>  ✅ aktif\n"
        f"  TP      : <code>{tp_r}</code>  ✅ aktif\n"
        f"  TP 30%  : <code>{partial_tp:.8f}</code>\n"
        f"  Leverage: <b>{leverage}x</b>\n"
        f"  Margin  : <code>{margin_used} USDT</code>\n"
        f"  ⭐ Special coin — filled langsung"
    )

    logger.info("[live] Market order %s %s fill=%.6f sl=%.6f tp=%.6f",
                symbol, side, fill_price, sl_r, tp_r)

    return {
        "ok":           True,
        "order_type":   "MARKET",
        "special_coin": True,
        "symbol":       symbol,
        "side":         side,
        "order_id":     order_id,
        "qty":          qty,
        "entry_price":  fill_price,
        "fill_price":   fill_price,
        "stop_loss":    sl_r,
        "take_profit":  tp_r,
        "leverage":     leverage,
        "balance_used": margin_used,
        "note":         "⭐ Special coin LIVE market order",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Breakeven: amend SL Binance ke entry (dipanggil monitor)
# ─────────────────────────────────────────────────────────────────────────────

def amend_breakeven_sl(pos: dict, notify_fn=None) -> dict:
    """
    Pindahkan SL Binance ke entry saat breakeven hit.
    Return updated pos dict dengan sl_order_id baru.
    """
    def _n(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    symbol      = pos.get("symbol")
    side        = pos.get("side", "BUY")
    entry       = float(pos.get("entry_price", 0))
    sl_order_id = pos.get("sl_order_id")

    if not sl_order_id:
        logger.warning("[live] amend_breakeven_sl: %s tidak ada sl_order_id", symbol)
        return pos

    from order.executor import amend_sl_to_price, get_tick_size
    tick     = get_tick_size(symbol)
    be_price = round(round(entry / tick) * tick, 8)

    new_sl_id = amend_sl_to_price(symbol, sl_order_id, side, be_price)
    if new_sl_id:
        pos_updated = dict(pos)
        pos_updated["sl_order_id"]  = new_sl_id
        pos_updated["breakeven_hit"] = True

        positions = _load_positions()
        for i, p in enumerate(positions):
            if p.get("order_id") == pos.get("order_id"):
                positions[i] = pos_updated
                break
        _save_positions(positions)

        logger.info("[live] Breakeven SL amended %s @ %.6f id=%s", symbol, be_price, new_sl_id)
        return pos_updated

    logger.error("[live] amend_breakeven_sl gagal %s", symbol)
    _n(f"🚨 <b>{symbol}</b> — Gagal pasang breakeven SL! Cek posisi manual.")
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Backward compat alias (dipakai scheduler/pipeline yang masih pakai nama lama)
# ─────────────────────────────────────────────────────────────────────────────

def execute_paper_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """Alias untuk execute_limit_order — backward compat."""
    return execute_limit_order(ai_result, pred, notify_fn=notify_fn)


def execute_paper_market_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """Alias untuk execute_market_order — backward compat."""
    return execute_market_order(ai_result, pred, notify_fn=notify_fn)


def has_paper_position(symbol: str) -> bool:
    """Alias untuk has_position — backward compat."""
    return has_position(symbol)
