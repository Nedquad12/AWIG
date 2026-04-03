# =============================================================
# order/paper_watcher.py — Monitor paper positions tiap 5 menit
# Cek candle high/low terhadap TP dan SL
# =============================================================

import json
import logging
import os
import threading
import time

import requests

from config import BINANCE_DATA_URL, DEFAULT_INTERVAL
from order.paper_executor import (
    PAPER_POSITIONS_FILE,
    PAPER_HISTORY_FILE,
    TAKER_FEE,
    _load_positions,
    _save_positions,
    _append_history,
    _load_history,
)

logger = logging.getLogger(__name__)

WATCH_INTERVAL_SEC = 5 * 60  # cek tiap 5 menit

_watcher_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ------------------------------------------------------------------
# Fetch candle terbaru (1 candle terakhir)
# ------------------------------------------------------------------

def _fetch_latest_candle(symbol: str, interval: str = DEFAULT_INTERVAL) -> dict | None:
    """
    Ambil 1 candle terbaru dari Binance public API.
    Return dict dengan open, high, low, close.
    """
    try:
        url = f"{BINANCE_DATA_URL}/fapi/v1/klines"
        resp = requests.get(url, params={
            "symbol":   symbol,
            "interval": interval,
            "limit":    2,  # ambil 2: index 0 = candle sebelumnya (closed), index 1 = live
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Pakai candle yang sudah closed (index 0) untuk hindari partial candle
        if not data or len(data) < 1:
            return None

        # Gunakan candle paling baru yang sudah closed
        c = data[-2] if len(data) >= 2 else data[-1]
        return {
            "open":  float(c[1]),
            "high":  float(c[2]),
            "low":   float(c[3]),
            "close": float(c[4]),
        }
    except Exception as e:
        logger.error("[paper_watcher] Gagal fetch candle %s: %s", symbol, e)
        return None


def _fetch_mark_price(symbol: str) -> float | None:
    """Ambil mark price saat ini untuk estimasi PnL realtime."""
    try:
        url  = f"{BINANCE_DATA_URL}/fapi/v1/premiumIndex"
        resp = requests.get(url, params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        return float(resp.json().get("markPrice", 0))
    except Exception as e:
        logger.warning("[paper_watcher] Gagal fetch mark price %s: %s", symbol, e)
        return None


# ------------------------------------------------------------------
# Evaluasi satu posisi
# ------------------------------------------------------------------

def _evaluate_position(pos: dict, candle: dict, notify_fn=None) -> str | None:
    """
    Evaluasi apakah TP atau SL sudah tersentuh di candle ini.
    Return: "TP", "SL", atau None jika belum.
    
    Logic:
    - LONG  (BUY):  TP hit jika high >= tp | SL hit jika low <= sl
    - SHORT (SELL): TP hit jika low  <= tp | SL hit jika high >= sl
    
    Jika dalam 1 candle both hit → konservatif: anggap SL dulu (worst case).
    """
    side = pos["side"]
    sl   = pos["stop_loss"]
    tp   = pos["take_profit"]
    high = candle["high"]
    low  = candle["low"]

    if side == "BUY":
        tp_hit = high >= tp
        sl_hit = low  <= sl
    else:  # SELL
        tp_hit = low  <= tp
        sl_hit = high >= sl

    if sl_hit and tp_hit:
        # Kedua hit dalam 1 candle → worst case = SL
        logger.warning(
            "[paper_watcher] %s %s — Both TP & SL hit in same candle → assume SL (worst case)",
            pos["symbol"], pos["paper_id"],
        )
        return "SL"
    elif tp_hit:
        return "TP"
    elif sl_hit:
        return "SL"
    return None


def _close_position(pos: dict, result: str, exit_price: float, notify_fn=None) -> dict:
    """
    Hitung PnL final dan simpan ke history.
    """
    def _notify(msg: str):
        if notify_fn:
            try:
                notify_fn(msg)
            except Exception:
                pass

    entry   = pos["entry_price"]
    side    = pos["side"]
    notional = pos["notional"]
    fee_open = pos["fee_open"]
    fee_close = round(notional * TAKER_FEE, 4)

    if side == "BUY":
        gross_pnl = (exit_price - entry) / entry * notional
    else:
        gross_pnl = (entry - exit_price) / entry * notional

    net_pnl = round(gross_pnl - fee_open - fee_close, 4)

    closed_record = {
        **pos,
        "status":     result,           # "TP" or "SL"
        "exit_price": exit_price,
        "pnl":        net_pnl,
        "fee_close":  fee_close,
        "closed_at":  int(time.time()),
    }

    _append_history(closed_record)

    duration_h = (closed_record["closed_at"] - pos["opened_at"]) / 3600
    pnl_emoji  = "✅" if net_pnl > 0 else "❌"
    result_emoji = "🎯" if result == "TP" else "🛑"

    _notify(
        f"{result_emoji} <b>PAPER TRADE CLOSED — {pos['symbol']}</b>\n"
        f"─────────────────────────\n"
        f"  Paper ID    : <code>{pos['paper_id']}</code>\n"
        f"  Side        : <b>{side}</b>\n"
        f"  Result      : <b>{result}</b>\n"
        f"  Entry       : <code>{entry}</code>\n"
        f"  Exit        : <code>{exit_price}</code>\n"
        f"  PnL (net)   : {pnl_emoji} <b>{net_pnl:+.4f} USDT</b>\n"
        f"  Duration    : <code>{duration_h:.1f} jam</code>\n"
        f"  <i>Fee total: {fee_open + fee_close:.4f} USDT</i>"
    )

    logger.info(
        "[paper_watcher] CLOSED %s %s — result=%s exit=%.6f pnl=%.4f USDT",
        pos["symbol"], pos["paper_id"], result, exit_price, net_pnl,
    )

    return closed_record


# ------------------------------------------------------------------
# Main watch loop
# ------------------------------------------------------------------

def _watch_loop(notify_fn=None):
    logger.info("[paper_watcher] Watcher started — interval=%ds", WATCH_INTERVAL_SEC)

    while not _stop_event.is_set():
        try:
            positions = _load_positions()
            open_positions = [p for p in positions if p.get("status") == "open"]

            if not open_positions:
                logger.debug("[paper_watcher] Tidak ada posisi aktif.")
            else:
                logger.info("[paper_watcher] Cek %d posisi aktif...", len(open_positions))

            closed_ids = []

            for pos in open_positions:
                symbol = pos["symbol"]
                candle = _fetch_latest_candle(symbol)
                if candle is None:
                    logger.warning("[paper_watcher] Skip %s — gagal fetch candle", symbol)
                    continue

                result = _evaluate_position(pos, candle, notify_fn)

                if result:
                    exit_price = pos["take_profit"] if result == "TP" else pos["stop_loss"]
                    _close_position(pos, result, exit_price, notify_fn)
                    closed_ids.append(pos["paper_id"])
                else:
                    # Belum closed — kirim update mark price
                    mark = _fetch_mark_price(symbol)
                    if mark:
                        entry = pos["entry_price"]
                        notional = pos["notional"]
                        side = pos["side"]
                        if side == "BUY":
                            unrealized = (mark - entry) / entry * notional
                        else:
                            unrealized = (entry - mark) / entry * notional
                        logger.debug(
                            "[paper_watcher] %s unrealized PnL: %.4f USDT (mark=%.6f)",
                            symbol, unrealized, mark,
                        )

            # Update posisi — hapus yang sudah closed
            if closed_ids:
                remaining = [p for p in positions if p.get("paper_id") not in closed_ids]
                _save_positions(remaining)

        except Exception as e:
            logger.error("[paper_watcher] Error di watch loop: %s", e, exc_info=True)

        # Tunggu 5 menit atau sampai stop
        _stop_event.wait(WATCH_INTERVAL_SEC)

    logger.info("[paper_watcher] Watcher stopped.")


# ------------------------------------------------------------------
# Start / stop watcher
# ------------------------------------------------------------------

def start_paper_watcher(notify_fn=None):
    global _watcher_thread, _stop_event

    if _watcher_thread and _watcher_thread.is_alive():
        logger.warning("[paper_watcher] Watcher sudah berjalan.")
        return

    _stop_event.clear()
    _watcher_thread = threading.Thread(
        target=_watch_loop,
        args=(notify_fn,),
        daemon=True,
        name="paper-watcher",
    )
    _watcher_thread.start()
    logger.info("[paper_watcher] Thread started: paper-watcher")


def stop_paper_watcher():
    global _stop_event
    _stop_event.set()
    logger.info("[paper_watcher] Stop signal sent.")


# ------------------------------------------------------------------
# Status helper — dipakai Telegram /paper_status
# ------------------------------------------------------------------

def get_paper_status() -> dict:
    """Return ringkasan posisi aktif + history untuk Telegram."""
    from config import PAPER_BALANCE_USDT
    positions = _load_positions()
    open_pos  = [p for p in positions if p.get("status") == "open"]
    history   = _load_history()

    total_trades = len(history)
    wins         = [h for h in history if h.get("status") == "TP"]
    losses       = [h for h in history if h.get("status") == "SL"]
    pnl_list     = [h.get("pnl", 0) for h in history]
    total_pnl    = sum(pnl_list)
    winrate      = len(wins) / total_trades * 100 if total_trades > 0 else 0
    best_trade   = max(pnl_list) if pnl_list else None
    worst_trade  = min(pnl_list) if pnl_list else None

    return {
        "open_positions":  open_pos,
        "total_trades":    total_trades,
        "wins":            len(wins),
        "losses":          len(losses),
        "winrate":         round(winrate, 1),
        "total_pnl":       round(total_pnl, 2),
        "best_trade":      round(best_trade, 2) if best_trade is not None else None,
        "worst_trade":     round(worst_trade, 2) if worst_trade is not None else None,
        "paper_balance":   PAPER_BALANCE_USDT,
        "history":         history[-10:],
    }


def format_paper_status_telegram(status: dict) -> str:
    """Format status paper trading untuk Telegram HTML — visual & informatif."""
    open_pos  = status["open_positions"]
    total     = status["total_trades"]
    wins      = status["wins"]
    losses    = status["losses"]
    winrate   = status["winrate"]
    total_pnl = status["total_pnl"]

    pnl_emoji = "📈" if total_pnl >= 0 else "📉"
    pnl_sign  = "+" if total_pnl >= 0 else ""

    lines = [
        "📋 <b>PAPER TRADING STATUS</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  💰 Virtual Balance : <b>${status.get('paper_balance', 5000):,.0f} USDT</b>",
        f"  {pnl_emoji} Total PnL (net)  : <b>{pnl_sign}{total_pnl:.2f} USDT</b>",
        f"  📊 W / L           : <b>{wins}✅  {losses}❌</b>  ({total} trades)",
        f"  🎯 Winrate         : <b>{winrate:.1f}%</b>",
    ]

    if status.get("best_trade") is not None:
        lines.append(f"  🏆 Best trade      : <b>+{status['best_trade']:.2f} USDT</b>")
    if status.get("worst_trade") is not None:
        lines.append(f"  💀 Worst trade     : <b>{status['worst_trade']:.2f} USDT</b>")

    # ── Posisi Aktif ───────────────────────────────────────────
    lines.append("")
    if open_pos:
        lines.append(f"<b>🔓 Posisi Aktif ({len(open_pos)}/4)</b>")
        lines.append("─────────────────────────")
        for p in open_pos:
            side_e     = "🟢" if p["side"] == "BUY" else "🔴"
            duration_m = (time.time() - p["opened_at"]) / 60
            dur_str    = f"{duration_m/60:.1f}h" if duration_m >= 60 else f"{duration_m:.0f}m"

            mark = _fetch_mark_price(p["symbol"])
            if mark:
                notional = p["notional"]
                if p["side"] == "BUY":
                    unreal = (mark - p["entry_price"]) / p["entry_price"] * notional
                else:
                    unreal = (p["entry_price"] - mark) / p["entry_price"] * notional
                unreal_net  = round(unreal - p["fee_open"] * 2, 2)
                unreal_sign = "+" if unreal_net >= 0 else ""
                unreal_e    = "🟢" if unreal_net >= 0 else "🔴"
                upnl_str    = f"{unreal_e} uPnL: <b>{unreal_sign}{unreal_net:.2f} USDT</b>  (mark {mark:.4f})"
            else:
                upnl_str = "uPnL: N/A"

            pnl_tp = p.get('pnl_if_tp', 0)
            pnl_sl = p.get('pnl_if_sl', 0)
            lines += [
                f"  {side_e} <b>{p['symbol']}</b>  ×{p['leverage']}  [{dur_str}]  <code>{p['paper_id']}</code>",
                f"     Entry <code>{p['entry_price']}</code>  →  SL <code>{p['stop_loss']}</code>  TP <code>{p['take_profit']}</code>",
                f"     {upnl_str}",
                f"     🎯 TP hit → <b>+{pnl_tp:.2f} USDT</b>  |  🛑 SL hit → <b>{pnl_sl:.2f} USDT</b>",
                "",
            ]
    else:
        lines.append("  Tidak ada posisi aktif.")
        lines.append("")

    # ── History 10 Terakhir ────────────────────────────────────
    if status["history"]:
        lines.append("<b>📜 10 Trade Terakhir</b>")
        lines.append("─────────────────────────")
        for h in reversed(status["history"]):
            result_e = "🎯" if h["status"] == "TP" else "🛑"
            pnl_s    = "+" if h["pnl"] >= 0 else ""
            side_lbl = "LONG" if h["side"] == "BUY" else "SHORT"
            lines.append(
                f"  {result_e} <b>{h['symbol']}</b> {side_lbl} → "
                f"<b>{pnl_s}{h['pnl']:.2f} USDT</b>"
            )

    return "\n".join(lines)
