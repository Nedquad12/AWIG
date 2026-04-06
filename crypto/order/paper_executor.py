"""
order/paper_executor.py — Modul utama order (paper & live).

Arsitektur:
  - Selalu dipanggil scheduler/pipeline, tidak peduli mode
  - Paper mode : balance dari JSON, order disimpan ke JSON saja
  - Live mode  : balance dari Binance (cache 35s), order dikirim ke Binance
                 DAN disalin ke JSON agar monitor bisa pantau
  - Logika bisnis (qty, notional, leverage clamp, re-entry, partial TP)
    ada di sini semua — executor.py hanya bridge Binance

paper_positions.json = satu-satunya sumber kebenaran posisi,
baik paper maupun live. monitor.py baca dari sini.

Proteksi berlapis (live):
  1. Volume Analyzer  → close paksa (via monitor)
  2. Trailing stop    → SL utama (via monitor, hajar Binance market)
  3. Breakeven SL     → SL Binance di entry (amend saat breakeven hit)
  4. SL/TP awal       → Binance order, fallback kalau sistem down

Partial TP (live):
  - monitor deteksi RR 1.5 via WS
  - kirim reduce-only market 30% ke Binance
  - update qty di JSON
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Callable, Optional

import requests

from config import (
    BINANCE_DATA_URL,
    PAPER_TRADING_MODE,
    PAPER_BALANCE_USDT,
    RISK_PER_TRADE_PCT,
)

logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────────────
_ROOT                = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_POSITIONS_FILE = os.path.join(_ROOT, "paper_positions.json")
PAPER_HISTORY_FILE   = os.path.join(_ROOT, "paper_history.json")

# ── Biaya & konstanta ─────────────────────────────────────────────────────────
TAKER_FEE          = 0.001
PARTIAL_TP_RR      = 1.5    # RR threshold partial TP
PARTIAL_TP_PCT     = 0.30   # tutup 30% posisi
REENTRY_MARGIN_CUT = 0.35   # kurangi 35% margin jika re-entry setelah TP


# ─────────────────────────────────────────────────────────────────────────────
# File I/O — positions & history
# ─────────────────────────────────────────────────────────────────────────────

def _load_positions() -> list:
    try:
        if os.path.exists(PAPER_POSITIONS_FILE):
            with open(PAPER_POSITIONS_FILE) as f:
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
            with open(PAPER_HISTORY_FILE) as f:
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


# ─────────────────────────────────────────────────────────────────────────────
# Balance helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_available_balance() -> float:
    """
    Paper mode : PAPER_BALANCE_USDT dikurangi margin open/pending.
    Live mode  : ambil dari Binance (cache 35s di executor.py).
    """
    if PAPER_TRADING_MODE:
        positions   = _load_positions()
        used_margin = sum(
            float(p.get("margin_used", 0))
            for p in positions
            if p.get("status") in ("open", "pending") and not p.get("live")
        )
        return round(max(PAPER_BALANCE_USDT - used_margin, 0), 4)
    else:
        from order.executor import get_available_balance as _live_balance
        return _live_balance()


# ─────────────────────────────────────────────────────────────────────────────
# Re-entry penalty
# ─────────────────────────────────────────────────────────────────────────────

def _get_reentry_multiplier(symbol: str) -> float:
    history   = _load_history()
    sym_hist  = [h for h in history if h.get("symbol") == symbol.upper()]
    if not sym_hist:
        return 1.0
    last = sorted(sym_hist, key=lambda x: x.get("closed_at", 0))[-1]
    if last.get("status") == "TP":
        logger.info("[paper] %s re-entry setelah TP → margin -%.0f%%",
                    symbol, REENTRY_MARGIN_CUT * 100)
        return 1.0 - REENTRY_MARGIN_CUT
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Symbol info & qty helpers (routing ke executor atau Binance public)
# ─────────────────────────────────────────────────────────────────────────────

def _get_sym_info(symbol: str) -> dict:
    from order.executor import get_symbol_info
    return get_symbol_info(symbol)


def _clamp_leverage(symbol: str, leverage: int) -> int:
    from order.executor import get_max_leverage
    max_lev = get_max_leverage(symbol)
    if leverage > max_lev:
        logger.info("[paper] %s leverage %dx > Binance max %dx → clamp",
                    symbol, leverage, max_lev)
        return max_lev
    return leverage


def _adjust_qty_to_min_notional(
    qty_fraction: float,
    available: float,
    leverage: int,
    entry_price: float,
    qty_step: float,
    min_notional: float,
    min_qty: float,
    max_qty: float,
) -> tuple[float, float, float]:
    """
    Naikkan qty_fraction sampai notional >= min_notional.
    Return (qty_fraction, qty, actual_notional).
    """
    from order.executor import round_step, MAX_NOTIONAL_USDT

    def _calc(frac):
        raw = available * frac * leverage
        cap = min(raw, MAX_NOTIONAL_USDT)
        qty = round_step(cap / entry_price, qty_step)
        qty = max(min_qty, min(qty, max_qty))
        return frac, qty, qty * entry_price

    _, qty, notional = _calc(qty_fraction)
    if notional >= min_notional:
        return qty_fraction, qty, notional

    needed = min((min_notional / (available * leverage)) * 1.01, 1.0) \
             if available > 0 and leverage > 0 else 1.0

    old_frac = qty_fraction
    _, qty, notional = _calc(needed)
    logger.info("[paper] min_notional adjust: fraction %.6f→%.6f notional %.4f→%.4f",
                old_frac, needed, old_frac * available * leverage, notional)
    return needed, qty, notional


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi umum: has_position, pending management
# ─────────────────────────────────────────────────────────────────────────────

def has_paper_position(symbol: str) -> bool:
    positions = _load_positions()
    return any(
        p["symbol"] == symbol.upper() and p.get("status") in ("open", "pending")
        for p in positions
    )


def activate_pending_position(paper_id: str, fill_price: float,
                               notify_fn=None) -> bool:
    """Ubah status pending → open saat harga hit entry (dipanggil monitor)."""
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    positions = _load_positions()
    for i, pos in enumerate(positions):
        if pos.get("paper_id") == paper_id and pos.get("status") == "pending":
            positions[i]["status"]    = "open"
            positions[i]["fill_price"] = fill_price
            positions[i]["filled_at"]  = int(time.time())
            _save_positions(positions)

            sym  = pos.get("symbol", "")
            side = pos.get("side", "BUY")
            emoji = "🟢" if side == "BUY" else "🔴"
            _notify(
                f"{emoji} <b>LIMIT FILLED — {sym}</b>\n"
                f"  ID     : <code>{paper_id}</code>\n"
                f"  Fill   : <code>{fill_price:.6f}</code>\n"
                f"  SL     : <code>{pos['stop_loss']}</code>\n"
                f"  TP     : <code>{pos['take_profit']}</code>\n"
                f"  Posisi aktif ✅"
            )
            logger.info("[paper] LIMIT FILLED %s %s @ %.6f", paper_id, sym, fill_price)
            return True

    logger.warning("[paper] activate_pending: %s tidak ditemukan", paper_id)
    return False


def cancel_pending_position(paper_id: str, reason: str = "expired",
                             notify_fn=None) -> bool:
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    positions = _load_positions()
    for i, pos in enumerate(positions):
        if pos.get("paper_id") == paper_id and pos.get("status") == "pending":
            positions[i]["status"]        = "cancelled"
            positions[i]["cancel_reason"] = reason
            positions[i]["cancelled_at"]  = int(time.time())
            _save_positions(positions)
            _notify(f"❌ <b>CANCELLED — {pos.get('symbol')}</b> ID: <code>{paper_id}</code> | {reason}")
            logger.info("[paper] CANCELLED %s %s", paper_id, reason)
            return True
    return False


def cancel_session_pending(session_id: str, notify_fn=None) -> list:
    """Cancel semua pending dari session_id (dipanggil monitor setelah 2 fills)."""
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    if not session_id:
        return []
    cancelled = []
    try:
        positions = _load_positions()
        changed   = False
        for i, pos in enumerate(positions):
            if pos.get("status") == "pending" and pos.get("session_id") == session_id:
                positions[i]["status"]        = "cancelled"
                positions[i]["cancel_reason"] = "sesi sudah 2 filled"
                positions[i]["cancelled_at"]  = int(time.time())
                changed = True
                pid = pos.get("paper_id", "?")
                sym = pos.get("symbol", "?")
                cancelled.append(pid)
                _notify(f"⚡ <b>Auto-Cancel — {sym}</b> ID: <code>{pid}</code> | sesi 2 filled")

                # Live: cancel juga di Binance kalau ada binance_order_id
                if not PAPER_TRADING_MODE and pos.get("binance_order_id"):
                    try:
                        from order.executor import cancel_order
                        cancel_order(sym, pos["binance_order_id"])
                    except Exception as e:
                        logger.warning("[paper] Gagal cancel Binance order %s: %s",
                                       pos.get("binance_order_id"), e)
        if changed:
            _save_positions(positions)
    except Exception as e:
        logger.error("[paper] cancel_session_pending error: %s", e)
    return cancelled


def count_session_filled(session_id: str) -> int:
    if not session_id:
        return 0
    try:
        return sum(
            1 for p in _load_positions()
            if p.get("session_id") == session_id and p.get("status") == "open"
        )
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Partial TP
# ─────────────────────────────────────────────────────────────────────────────

def check_partial_tp(pos: dict, current_price: float) -> bool:
    if pos.get("partial_tp_done"):
        return False
    entry = float(pos.get("entry_price", 0))
    sl    = float(pos.get("sl_initial", pos.get("stop_loss", 0)))
    side  = pos.get("side", "BUY")
    if entry <= 0 or sl <= 0:
        return False
    risk = abs(entry - sl)
    if risk <= 0:
        return False
    if side == "BUY":
        return current_price >= entry + PARTIAL_TP_RR * risk
    else:
        return current_price <= entry - PARTIAL_TP_RR * risk


def execute_partial_tp(pos: dict, current_price: float,
                        notify_fn=None) -> dict:
    """
    Close 30% posisi saat RR 1.5:1.
    Live: kirim reduce-only market order ke Binance.
    Kedua mode: update qty di JSON.
    Return updated position dict.
    """
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    symbol       = pos.get("symbol", "")
    side         = pos.get("side", "BUY")
    partial_qty  = round(float(pos.get("qty", 0)) * PARTIAL_TP_PCT, 6)
    partial_notional = float(pos.get("notional", 0)) * PARTIAL_TP_PCT
    entry        = float(pos.get("entry_price", 0))
    is_live      = pos.get("live", False)

    if side == "BUY":
        partial_pnl = (current_price - entry) / entry * partial_notional
    else:
        partial_pnl = (entry - current_price) / entry * partial_notional
    partial_pnl -= partial_notional * TAKER_FEE * 2
    partial_pnl  = round(partial_pnl, 4)

    # Live: kirim ke Binance
    if is_live and not PAPER_TRADING_MODE:
        try:
            from order.executor import close_position_market
            close_position_market(symbol, side, partial_qty)
            logger.info("[paper] Live partial TP %s %.6f qty @ %.6f",
                        symbol, partial_qty, current_price)
        except Exception as e:
            logger.error("[paper] Gagal partial TP Binance %s: %s", symbol, e)
            _notify(f"🚨 <b>{symbol}</b> — Partial TP Binance gagal: <code>{e}</code>")
            return pos  # jangan update JSON jika Binance gagal

    # Update posisi di JSON
    new_qty      = round(float(pos.get("qty", 0)) * (1 - PARTIAL_TP_PCT), 6)
    new_notional = round(float(pos.get("notional", 0)) * (1 - PARTIAL_TP_PCT), 4)
    new_margin   = round(new_notional / float(pos.get("leverage", 1)), 4)

    pos_updated = dict(pos)
    pos_updated.update({
        "qty":              new_qty,
        "notional":         new_notional,
        "margin_used":      new_margin,
        "partial_tp_done":  True,
        "partial_tp_price": current_price,
        "partial_tp_pnl":   partial_pnl,
    })

    # Simpan partial ke history
    partial_record = dict(pos)
    partial_record.update({
        "status":           "PARTIAL_TP",
        "close_reason":     f"Partial TP {PARTIAL_TP_PCT*100:.0f}% @ RR {PARTIAL_TP_RR}",
        "close_price":      current_price,
        "pnl":              partial_pnl,
        "closed_at":        time.time(),
        "qty_closed":       partial_qty,
        "notional_closed":  round(partial_notional, 4),
    })
    _append_history(partial_record)

    # Update positions file
    positions = _load_positions()
    for i, p in enumerate(positions):
        if p.get("paper_id") == pos.get("paper_id"):
            positions[i] = pos_updated
            break
    _save_positions(positions)

    pnl_str = f"+{partial_pnl:.2f}" if partial_pnl >= 0 else f"{partial_pnl:.2f}"
    mode_tag = "LIVE" if is_live else "PAPER"
    _notify(
        f"🎯 <b>Partial TP {PARTIAL_TP_PCT*100:.0f}% [{mode_tag}] — {symbol}</b>\n"
        f"  Price  : <code>{current_price:.6f}</code>  (RR {PARTIAL_TP_RR}:1)\n"
        f"  PnL    : <b>{pnl_str} USDT</b>\n"
        f"  Sisa   : {(1-PARTIAL_TP_PCT)*100:.0f}% posisi masih aktif\n"
        f"  SL geser ke entry (breakeven) ✅"
    )
    return pos_updated


# ─────────────────────────────────────────────────────────────────────────────
# Breakeven: amend SL Binance ke entry (live only)
# ─────────────────────────────────────────────────────────────────────────────

def amend_breakeven_sl(pos: dict, notify_fn=None) -> dict:
    """
    Pindahkan SL Binance ke harga entry saat breakeven hit.
    Update sl_order_id baru di JSON.
    Hanya aktif di live mode.
    Return updated pos dict.
    """
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    if PAPER_TRADING_MODE or not pos.get("live"):
        return pos

    symbol      = pos.get("symbol")
    side        = pos.get("side", "BUY")
    entry       = float(pos.get("entry_price", 0))
    sl_order_id = pos.get("sl_order_id")

    if not sl_order_id:
        logger.warning("[paper] amend_breakeven_sl: %s tidak ada sl_order_id", symbol)
        return pos

    from order.executor import amend_sl_to_price, round_price, get_tick_size
    tick     = get_tick_size(symbol)
    be_price = round_price(entry, tick)

    new_sl_id = amend_sl_to_price(symbol, sl_order_id, side, be_price)
    if new_sl_id:
        pos_updated = dict(pos)
        pos_updated["sl_order_id"] = new_sl_id
        pos_updated["breakeven_hit"] = True

        positions = _load_positions()
        for i, p in enumerate(positions):
            if p.get("paper_id") == pos.get("paper_id"):
                positions[i] = pos_updated
                break
        _save_positions(positions)

        _notify(
            f"⚖️ <b>Breakeven SL — {symbol}</b>\n"
            f"  SL Binance dipindah ke entry: <code>{be_price}</code>\n"
            f"  Trailing stop aktif ✅ (proteksi ganda)"
        )
        logger.info("[paper] Breakeven SL amended %s @ %.6f new_sl_id=%s",
                    symbol, be_price, new_sl_id)
        return pos_updated

    logger.error("[paper] amend_breakeven_sl gagal %s — SL lama sudah di-cancel!", symbol)
    _notify(f"🚨 <b>{symbol}</b> — Gagal pasang breakeven SL! Cek posisi manual.")
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Background thread: poll Binance fill + pasang SL/TP awal (live only)
# ─────────────────────────────────────────────────────────────────────────────

def _live_fill_watcher(
    paper_id:   str,
    symbol:     str,
    order_id:   int,
    side:       str,
    sl_price:   float,
    tp_price:   float,
    qty:        float,
    notify_fn:  Optional[Callable],
):
    """
    Background thread untuk live mode.
    Poll Binance sampai fill → pasang SL/TP → update JSON.
    Fill monitoring via Binance (bukan WS) — WS monitor.py tetap jalan paralel.
    """
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    from order.executor import (
        poll_until_filled, place_stop_market, place_take_profit_market,
        cancel_order, close_position_market, get_mark_price,
        round_price, get_tick_size, cancel_all_open_orders,
    )

    logger.info("[paper:bg] Watching fill %s order_id=%d", symbol, order_id)

    filled_order = poll_until_filled(symbol, order_id, timeout_sec=1200)

    if filled_order is None:
        # Timeout / cancel: hapus dari JSON
        _cancel_live_position(paper_id, symbol, "timeout — tidak ter-fill 20 menit")
        _notify(f"⏱ <b>{symbol}</b> — Entry timeout, order di-cancel.")
        return

    filled_price = float(filled_order.get("avgPrice", sl_price))
    filled_qty   = float(filled_order.get("executedQty", qty))

    # Update status open di JSON
    positions = _load_positions()
    for i, pos in enumerate(positions):
        if pos.get("paper_id") == paper_id:
            positions[i]["status"]    = "open"
            positions[i]["fill_price"] = filled_price
            positions[i]["filled_at"]  = int(time.time())
            break
    _save_positions(positions)

    logger.info("[paper:bg] %s FILLED @ %.6f qty=%.6f", symbol, filled_price, filled_qty)

    # Cek mark price — jangan pasang SL yang sudah terlewat
    mark = get_mark_price(symbol)
    if mark is None:
        mark = filled_price

    tick     = get_tick_size(symbol)
    sl_side  = "SELL" if side == "BUY" else "BUY"
    tp_side  = "SELL" if side == "BUY" else "BUY"
    sl_r     = round_price(sl_price, tick)
    tp_r     = round_price(tp_price, tick)

    # Validasi SL tidak sudah terlewat mark
    sl_invalid = (side == "BUY" and sl_r >= mark) or (side == "SELL" and sl_r <= mark)
    if sl_invalid:
        logger.warning("[paper:bg] %s SL %.6f sudah terlewat mark %.6f — emergency close",
                       symbol, sl_r, mark)
        try:
            close_position_market(symbol, side, filled_qty)
        except Exception as e:
            logger.error("[paper:bg] Emergency close gagal: %s", e)
        _notify(f"🚨 <b>{symbol}</b> — SL sudah terlewat mark price, emergency close.")
        _cancel_live_position(paper_id, symbol, "SL sudah terlewat setelah fill")
        return

    # Pasang SL ke Binance
    sl_order_id = None
    try:
        sl_resp     = place_stop_market(symbol, sl_side, sl_r, close_position=True)
        sl_order_id = sl_resp.get("orderId")
        logger.info("[paper:bg] SL placed %s id=%s", symbol, sl_order_id)
    except Exception as e:
        logger.error("[paper:bg] SL gagal %s: %s — emergency close", symbol, e)
        try:
            close_position_market(symbol, side, filled_qty)
        except Exception:
            pass
        _notify(f"🚨 <b>{symbol}</b> — SL gagal dipasang: <code>{e}</code>")
        _cancel_live_position(paper_id, symbol, f"SL gagal: {e}")
        return

    # Pasang TP ke Binance
    tp_order_id = None
    try:
        tp_resp     = place_take_profit_market(symbol, tp_side, tp_r, close_position=True)
        tp_order_id = tp_resp.get("orderId")
        logger.info("[paper:bg] TP placed %s id=%s", symbol, tp_order_id)
    except Exception as e:
        logger.error("[paper:bg] TP gagal %s: %s — cancel SL + emergency close", symbol, e)
        if sl_order_id:
            cancel_order(symbol, sl_order_id)
        try:
            close_position_market(symbol, side, filled_qty)
        except Exception:
            pass
        _notify(f"🚨 <b>{symbol}</b> — TP gagal dipasang: <code>{e}</code>")
        _cancel_live_position(paper_id, symbol, f"TP gagal: {e}")
        return

    # Update sl_order_id & tp_order_id di JSON
    positions = _load_positions()
    for i, pos in enumerate(positions):
        if pos.get("paper_id") == paper_id:
            positions[i]["sl_order_id"] = sl_order_id
            positions[i]["tp_order_id"] = tp_order_id
            positions[i]["fill_price"]  = filled_price
            positions[i]["qty"]         = filled_qty
            break
    _save_positions(positions)

    side_emoji = "🟢" if side == "BUY" else "🔴"
    _notify(
        f"{side_emoji} <b>SL & TP Aktif [LIVE] — {symbol}</b>\n"
        f"  Fill   : <code>{filled_price}</code>\n"
        f"  SL     : <code>{sl_r}</code>  id={sl_order_id} ✅\n"
        f"  TP     : <code>{tp_r}</code>  id={tp_order_id} ✅\n"
        f"  Monitor aktif via WS + Binance order"
    )


def _cancel_live_position(paper_id: str, symbol: str, reason: str) -> None:
    """Tandai posisi live sebagai cancelled di JSON."""
    positions = _load_positions()
    for i, pos in enumerate(positions):
        if pos.get("paper_id") == paper_id:
            positions[i]["status"]        = "cancelled"
            positions[i]["cancel_reason"] = reason
            positions[i]["cancelled_at"]  = int(time.time())
            break
    _save_positions(positions)
    logger.info("[paper] Live position cancelled %s %s: %s", paper_id, symbol, reason)


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi publik utama — dipanggil scheduler/pipeline
# ─────────────────────────────────────────────────────────────────────────────

def execute_paper_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """
    Entry point utama untuk LIMIT order (paper & live).

    Paper mode : catat ke JSON, monitor pantau via WS aggTrade.
    Live mode  : kirim LIMIT ke Binance, salin ke JSON,
                 bg-thread poll fill lalu pasang SL/TP.
    """
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    from order.executor import (
        get_symbol_info, set_leverage, place_limit_order,
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

    # Re-entry penalty
    reentry_mult = _get_reentry_multiplier(symbol)
    if reentry_mult < 1.0:
        qty_fraction = max(0.001, round(qty_fraction * reentry_mult, 6))

    # Clamp leverage ke max Binance
    leverage = _clamp_leverage(symbol, leverage)

    # Ambil symbol info dari Binance (akurat untuk qty/notional)
    try:
        sym_info = get_symbol_info(symbol)
    except Exception as e:
        return {"ok": False, "symbol": symbol, "reason_fail": f"symbol_info error: {e}"}

    # Balance
    available = get_available_balance()
    if available <= 0:
        return {"ok": False, "symbol": symbol,
                "reason_fail": f"Balance tidak cukup: {available:.2f} USDT"}

    # Hitung qty
    raw_notional    = available * qty_fraction * leverage
    capped_notional = min(raw_notional, MAX_NOTIONAL_USDT)
    qty = round_step(capped_notional / entry_price, sym_info["qty_step"])

    entry_r = round_price(entry_price, sym_info["price_tick"])
    sl_r    = round_price(stop_loss,   sym_info["price_tick"])
    tp_r    = round_price(take_profit, sym_info["price_tick"])

    # Adjust ke min_notional jika perlu
    actual_notional = qty * entry_r
    if actual_notional < sym_info["min_notional"]:
        qty_fraction, qty, actual_notional = _adjust_qty_to_min_notional(
            qty_fraction, available, leverage, entry_r,
            sym_info["qty_step"], sym_info["min_notional"],
            sym_info["min_qty"],  sym_info["max_qty"],
        )

    if qty < sym_info["min_qty"]:
        return {"ok": False, "symbol": symbol,
                "reason_fail": (f"Qty {qty} < minQty {sym_info['min_qty']}. "
                                f"Balance={available:.2f} USDT")}

    if qty > sym_info["max_qty"]:
        qty = sym_info["max_qty"]

    actual_notional = qty * entry_r
    if actual_notional < sym_info["min_notional"]:
        return {"ok": False, "symbol": symbol,
                "reason_fail": (f"Notional {actual_notional:.4f} < min "
                                f"{sym_info['min_notional']} USDT. Balance terlalu kecil.")}

    margin_used = round(actual_notional / leverage, 4)

    # Partial TP price
    risk = abs(entry_r - sl_r)
    partial_tp_price = round(
        (entry_r + PARTIAL_TP_RR * risk) if side == "BUY"
        else (entry_r - PARTIAL_TP_RR * risk),
        8
    )
    rr = round(abs(tp_r - entry_r) / risk, 2) if risk > 0 else 0

    paper_id   = str(uuid.uuid4())[:12]
    now_ts     = int(time.time())
    is_live    = not PAPER_TRADING_MODE

    # ── Buat record posisi ─────────────────────────────────────────────
    position = {
        "paper_id":         paper_id,
        "live":             is_live,
        "status":           "pending",
        "symbol":           symbol,
        "side":             side,
        "entry_price":      entry_r,
        "stop_loss":        sl_r,
        "sl_initial":       sl_r,       # untuk trailing & partial TP calc
        "take_profit":      tp_r,
        "partial_tp_price": partial_tp_price,
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
        "binance_order_id": None,
        "sl_order_id":      None,
        "tp_order_id":      None,
    }

    # ── Live: kirim LIMIT ke Binance ───────────────────────────────────
    if is_live:
        try:
            leverage = set_leverage(symbol, leverage)  # set ke Binance
            resp     = place_limit_order(symbol, side, qty, entry_r)
            binance_order_id = resp.get("orderId")
        except Exception as e:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"Binance LIMIT order gagal: {e}"}

        position["binance_order_id"] = binance_order_id
        position["leverage"]         = leverage  # post-clamp

        # Simpan ke JSON dulu (status pending)
        positions = _load_positions()
        positions.append(position)
        _save_positions(positions)

        # Background thread: poll fill → pasang SL/TP
        t = threading.Thread(
            target=_live_fill_watcher,
            args=(paper_id, symbol, binance_order_id, side,
                  sl_r, tp_r, qty, notify_fn),
            daemon=True,
            name=f"fill-{symbol}-{paper_id}",
        )
        t.start()

    else:
        # Paper: simpan langsung
        positions = _load_positions()
        positions.append(position)
        _save_positions(positions)

    # ── Notifikasi ─────────────────────────────────────────────────────
    pnl_tp = round(
        (tp_r - entry_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2
        if side == "BUY" else
        (entry_r - tp_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2,
        4
    )
    pnl_sl = round(
        (sl_r - entry_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2
        if side == "BUY" else
        (entry_r - sl_r) / entry_r * actual_notional - actual_notional * TAKER_FEE * 2,
        4
    )

    mode_tag   = "LIVE" if is_live else "PAPER"
    side_emoji = "🟢" if side == "BUY" else "🔴"
    pnl_tp_str = f"+{pnl_tp:.2f}" if pnl_tp > 0 else f"{pnl_tp:.2f}"

    _notify(
        f"📋 <b>{mode_tag} LIMIT ORDER — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  {side_emoji} <b>{side}</b>  ×{leverage}  |  ID: <code>{paper_id}</code>\n"
        f"  Entry    : <code>{entry_r}</code>  <i>(menunggu hit)</i>\n"
        f"  SL       : <code>{sl_r}</code>\n"
        f"  TP Full  : <code>{tp_r}</code>  (RR {rr})\n"
        f"  TP 30%   : <code>{partial_tp_price}</code>  (RR {PARTIAL_TP_RR}:1)\n"
        f"  Notional : <code>{actual_notional:.2f} USDT</code>  margin: <code>{margin_used} USDT</code>\n"
        f"  WTI      : <code>{wti_pct:.1f}%</code>\n"
        f"  Est TP   : <b>{pnl_tp_str} USDT</b>  |  Est SL: <b>{pnl_sl:.2f} USDT</b>"
        + (f"\n  ⚠️ Re-entry: margin -{REENTRY_MARGIN_CUT*100:.0f}%"
           if reentry_mult < 1.0 else "")
        + (f"\n  💰 Saldo: <b>{available:.2f} USDT</b>" if not is_live else "")
    )

    return {
        "ok":           True,
        "paper":        not is_live,
        "live":         is_live,
        "symbol":       symbol,
        "side":         side,
        "order_id":     paper_id,
        "qty":          qty,
        "entry_price":  entry_r,
        "stop_loss":    sl_r,
        "take_profit":  tp_r,
        "leverage":     leverage,
        "balance_used": margin_used,
        "notional":     round(actual_notional, 4),
        "qty_fraction": round(qty_fraction, 6),
        "wti_pct":      wti_pct,
        "note":         f"{'LIVE LIMIT — fill dipantau Binance + WS' if is_live else 'PAPER LIMIT — entry dipantau via WS'}",
    }


def execute_paper_market_order(ai_result: dict, pred: dict, notify_fn=None) -> dict:
    """
    MARKET order (special coin, fill langsung).
    Paper: fill @ mark price sekarang, status langsung open.
    Live : kirim MARKET ke Binance, pasang SL/TP langsung.
    """
    def _notify(msg):
        if notify_fn:
            try: notify_fn(msg)
            except Exception: pass

    from order.executor import (
        get_symbol_info, set_leverage, place_market_order,
        place_stop_market, place_take_profit_market,
        cancel_order, close_position_market,
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

    # Mark price (paper pakai ini sebagai fill price, live eksekusi market)
    mark_price = get_mark_price(symbol)
    if not mark_price:
        return {"ok": False, "symbol": symbol,
                "reason_fail": "Gagal ambil mark price"}

    try:
        sym_info = get_symbol_info(symbol)
    except Exception as e:
        return {"ok": False, "symbol": symbol, "reason_fail": f"symbol_info error: {e}"}

    reentry_mult = _get_reentry_multiplier(symbol)
    if reentry_mult < 1.0:
        qty_fraction = max(0.001, round(qty_fraction * reentry_mult, 6))

    available = get_available_balance()
    if available <= 0:
        return {"ok": False, "symbol": symbol,
                "reason_fail": f"Balance tidak cukup: {available:.2f}"}

    raw_notional    = available * qty_fraction * leverage
    capped_notional = min(raw_notional, MAX_NOTIONAL_USDT)
    qty = round_step(capped_notional / mark_price, sym_info["qty_step"])

    sl_r  = round_price(stop_loss,  sym_info["price_tick"])
    tp_r  = round_price(take_profit, sym_info["price_tick"])
    risk  = abs(mark_price - sl_r)
    partial_tp_price = round(
        (mark_price + PARTIAL_TP_RR * risk) if side == "BUY"
        else (mark_price - PARTIAL_TP_RR * risk), 8
    )

    margin_used     = round(capped_notional / leverage, 4)
    actual_notional = qty * mark_price

    paper_id = str(uuid.uuid4())[:12]
    now_ts   = int(time.time())
    is_live  = not PAPER_TRADING_MODE

    # ── Live: kirim ke Binance ─────────────────────────────────────────
    sl_order_id = None
    tp_order_id = None
    binance_order_id = None

    if is_live:
        try:
            leverage = set_leverage(symbol, leverage)
            mkt_resp = place_market_order(symbol, side, qty)
            binance_order_id = mkt_resp.get("orderId")
            fill_price       = float(mkt_resp.get("avgPrice", mark_price)) or mark_price
        except Exception as e:
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"Market order gagal: {e}"}

        sl_side = "SELL" if side == "BUY" else "BUY"
        tp_side = sl_side
        try:
            sl_resp     = place_stop_market(symbol, sl_side, sl_r, close_position=True)
            sl_order_id = sl_resp.get("orderId")
        except Exception as e:
            logger.error("[paper:market] SL gagal %s: %s", symbol, e)
            try: close_position_market(symbol, side, qty)
            except Exception: pass
            _notify(f"🚨 <b>{symbol}</b> — MARKET SL gagal: <code>{e}</code>")
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"SL gagal setelah market fill: {e}"}

        try:
            tp_resp     = place_take_profit_market(symbol, tp_side, tp_r, close_position=True)
            tp_order_id = tp_resp.get("orderId")
        except Exception as e:
            logger.error("[paper:market] TP gagal %s: %s", symbol, e)
            cancel_order(symbol, sl_order_id)
            try: close_position_market(symbol, side, qty)
            except Exception: pass
            _notify(f"🚨 <b>{symbol}</b> — MARKET TP gagal: <code>{e}</code>")
            return {"ok": False, "symbol": symbol,
                    "reason_fail": f"TP gagal setelah market fill: {e}"}
    else:
        fill_price = mark_price

    position = {
        "paper_id":         paper_id,
        "live":             is_live,
        "status":           "open",
        "symbol":           symbol,
        "side":             side,
        "entry_price":      fill_price,
        "fill_price":       fill_price,
        "stop_loss":        sl_r,
        "sl_initial":       sl_r,
        "take_profit":      tp_r,
        "partial_tp_price": partial_tp_price,
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
        "binance_order_id": binance_order_id,
        "sl_order_id":      sl_order_id,
        "tp_order_id":      tp_order_id,
    }

    positions = _load_positions()
    positions.append(position)
    _save_positions(positions)

    mode_tag   = "LIVE" if is_live else "PAPER"
    side_emoji = "🟢" if side == "BUY" else "🔴"
    sl_tp_note = (f"\n  SL id  : <code>{sl_order_id}</code> ✅"
                  f"\n  TP id  : <code>{tp_order_id}</code> ✅") if is_live else ""

    _notify(
        f"{side_emoji} <b>{mode_tag} MARKET ORDER — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  ID      : <code>{paper_id}</code>\n"
        f"  Fill    : <code>{fill_price}</code>\n"
        f"  SL      : <code>{sl_r}</code>\n"
        f"  TP      : <code>{tp_r}</code>\n"
        f"  TP 30%  : <code>{partial_tp_price:.8f}</code>\n"
        f"  Leverage: <b>{leverage}x</b>\n"
        f"  Qty     : <code>{qty}</code>\n"
        f"  Margin  : <code>{margin_used} USDT</code>\n"
        f"  ⭐ Special coin — market order langsung"
        + sl_tp_note
    )

    return {
        "ok":           True,
        "paper":        not is_live,
        "live":         is_live,
        "order_type":   "MARKET",
        "special_coin": True,
        "symbol":       symbol,
        "side":         side,
        "order_id":     paper_id,
        "qty":          qty,
        "entry_price":  fill_price,
        "fill_price":   fill_price,
        "stop_loss":    sl_r,
        "take_profit":  tp_r,
        "leverage":     leverage,
        "balance_used": margin_used,
        "note":         f"⭐ Special coin {'LIVE' if is_live else 'PAPER'} market order",
    }
