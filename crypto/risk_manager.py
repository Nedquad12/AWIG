import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_HISTORY_DIR    = "/home/ec2-user/crypto/history"
_STATE_FILE     = "/home/ec2-user/crypto/ban/risk_state.json" 
_PAPER_POS_FILE = "/home/ec2-user/crypto/positions.json"
_PAPER_HIST_FILE= "/home/ec2-user/crypto/positions_history.json"

os.makedirs(_HISTORY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)

SL_BAN_SESSIONS       = 2      
CIRCUIT_BREAKER_PCT   = 0.05   
CIRCUIT_BREAKER_SESS  = 7      
WTI_CORR_THRESHOLD    = 65.0   
MAX_CORR_POSITIONS    = 5      
BTC_SPIKE_DELAY_SEC   = 2 * 3600  

def _load_state() -> dict:
    try:
        if os.path.exists(_STATE_FILE):
            with open(_STATE_FILE) as f:
                return json.load(f)
    except Exception as e:
        logger.error("[risk] Load state error: %s", e)
    return {
        "sl_bans":             {},    # {symbol: sessions_remaining}
        "cb_sessions_left":    0,     # sesi circuit breaker tersisa
        "cb_triggered_at":     None,  # ISO timestamp saat CB triggered
        "session_count":       0,     # total sesi yang sudah berjalan
        "btc_spike_until":     0.0,   # epoch timestamp BTC spike cooldown
        "urgent_cb_date":      "",    # tanggal UTC saat urgent CB trigger (YYYY-MM-DD)
        "urgent_cb_direction": "",    # "UP" atau "DOWN"
        "urgent_cb_banned_side": "",  # "LONG" atau "SHORT" — di-ban 1 sesi
        "urgent_cb_sessions_left": 0, # sisa sesi ban (1 sesi) 
    }


def _save_state(state: dict) -> None:
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error("[risk] Save state error: %s", e)


def tick_session() -> dict:
    """
    Dipanggil di awal setiap sesi scan.
    Decrement semua counter, return state terbaru.
    """
    state = _load_state()
    state["session_count"] = state.get("session_count", 0) + 1

    bans = state.get("sl_bans", {})
    expired = [sym for sym, rem in bans.items() if rem <= 1]
    for sym in expired:
        del bans[sym]
        logger.info("[risk] Ban expired: %s", sym)
    for sym in bans:
        bans[sym] -= 1
    state["sl_bans"] = bans

    if state.get("cb_sessions_left", 0) > 0:
        state["cb_sessions_left"] -= 1
        logger.info("[risk] Circuit breaker: %d sesi tersisa", state["cb_sessions_left"])

    # Decrement urgent CB ban (1 sesi)
    if state.get("urgent_cb_sessions_left", 0) > 0:
        state["urgent_cb_sessions_left"] -= 1
        if state["urgent_cb_sessions_left"] == 0:
            logger.info("[risk] Urgent CB ban expired — %s diizinkan kembali",
                        state.get("urgent_cb_banned_side", ""))
            state["urgent_cb_banned_side"] = ""

    _save_state(state)
    return state


def register_sl(symbol: str) -> None:
    """Panggil setelah posisi kena SL. Ban simbol N sesi ke depan."""
    state = _load_state()
    bans  = state.get("sl_bans", {})
    sym   = symbol.upper()
    bans[sym] = SL_BAN_SESSIONS
    state["sl_bans"] = bans
    _save_state(state)
    logger.info("[risk] SL ban: %s selama %d sesi", sym, SL_BAN_SESSIONS)


def is_banned(symbol: str) -> Tuple[bool, int]:
    """Return (banned, sessions_remaining)."""
    state = _load_state()
    rem   = state.get("sl_bans", {}).get(symbol.upper(), 0)
    return rem > 0, rem

def check_circuit_breaker() -> Tuple[bool, float]:
    """
    Cek apakah daily loss >= CIRCUIT_BREAKER_PCT dari modal.
    Return (cb_active, loss_pct).
    """
    state = _load_state()

    if state.get("cb_sessions_left", 0) > 0:
        return True, 0.0

    try:
        from order.executor import get_available_balance as _live_bal
        modal_basis = _live_bal()
        if modal_basis <= 0:
            modal_basis = 350.0
    except Exception:
        modal_basis = 350.0

    today_pnl = _get_today_pnl()
    loss_pct  = (-today_pnl) / modal_basis

    if loss_pct >= CIRCUIT_BREAKER_PCT:
        state["cb_sessions_left"] = CIRCUIT_BREAKER_SESS
        state["cb_triggered_at"]  = datetime.now(timezone.utc).isoformat()
        _save_state(state)
        logger.warning(
            "[risk] ⚡ CIRCUIT BREAKER TRIGGERED! Loss=%.2f%% >= %.0f%% — "
            "stop %d sesi",
            loss_pct * 100, CIRCUIT_BREAKER_PCT * 100, CIRCUIT_BREAKER_SESS,
        )
        return True, loss_pct

    return False, loss_pct

def _get_today_pnl() -> float:
    """Hitung total PnL dari trade yang ditutup hari ini (UTC)."""
    try:
        if not os.path.exists(_PAPER_HIST_FILE):
            return 0.0
        with open(_PAPER_HIST_FILE) as f:
            history = json.load(f)
        today = datetime.now(timezone.utc).date()
        total = 0.0
        for h in history:
            closed_at = h.get("closed_at")
            if closed_at:
                closed_date = datetime.fromtimestamp(closed_at, tz=timezone.utc).date()
                if closed_date == today:
                    total += float(h.get("pnl", 0))
        return total
    except Exception as e:
        logger.warning("[risk] Gagal hitung today PnL: %s", e)
        return 0.0


def get_cb_state() -> dict:
    state = _load_state()
    return {
        "active":      state.get("cb_sessions_left", 0) > 0,
        "sessions_left": state.get("cb_sessions_left", 0),
        "triggered_at":  state.get("cb_triggered_at"),
        "today_pnl":     _get_today_pnl(),
    }
    
def register_btc_spike_cb(reason: str = "") -> float:
    """
    Dipanggil oleh monitor.py saat volume BTCUSDT buy/sell >= 2× ATR.
    Set btc_spike_until = now + BTC_SPIKE_DELAY_SEC (2 jam).
    Return timestamp epoch saat cooldown berakhir.
    """
    import time as _time
    state     = _load_state()
    until     = _time.time() + BTC_SPIKE_DELAY_SEC
    state["btc_spike_until"]        = until
    state["btc_spike_triggered_at"] = datetime.now(timezone.utc).isoformat()
    state["btc_spike_reason"]       = reason
    _save_state(state)
    logger.warning(
        "[risk] 🌊 BTC Spike CB aktif — scanner di-block 2 jam | reason: %s", reason
    )
    return until


def is_btc_spike_cooldown() -> tuple:
    import time as _time
    state = _load_state()
    until = float(state.get("btc_spike_until", 0.0))
    now   = _time.time()
    if now < until:
        return True, until - now, state.get("btc_spike_reason", "")
    return False, 0.0, ""


def count_correlated_positions() -> int:
    """Hitung berapa posisi aktif yang WTI >= WTI_CORR_THRESHOLD."""
    try:
        if not os.path.exists(_PAPER_POS_FILE):
            return 0
        with open(_PAPER_POS_FILE) as f:
            positions = json.load(f)
        open_pos = [p for p in positions if p.get("status") == "open"]
        return sum(
            1 for p in open_pos
            if float(p.get("wti_pct", 0)) >= WTI_CORR_THRESHOLD
        )
    except Exception as e:
        logger.warning("[risk] Gagal hitung correlated positions: %s", e)
        return 0
    
def register_urgent_cb(direction: str, banned_side: str) -> None:
    state = _load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    state["urgent_cb_date"]          = today
    state["urgent_cb_direction"]     = direction.upper()
    state["urgent_cb_banned_side"]   = banned_side.upper()
    state["urgent_cb_sessions_left"] = 1
    _save_state(state)
    logger.warning(
        "[risk] 🚨 Urgent CB registered — BTC %s | banned: %s | date: %s",
        direction, banned_side, today,
    )


def is_urgent_cb_triggered() -> tuple:
    """
    Cek apakah Urgent CB sudah trigger hari ini (candle 1D yang sama).
    Return (triggered: bool, direction: str)
    """
    state = _load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("urgent_cb_date") == today:
        return True, state.get("urgent_cb_direction", "")
    return False, ""


def get_urgent_cb_ban() -> tuple:
    state      = _load_state()
    banned_side = state.get("urgent_cb_banned_side", "")
    sessions   = state.get("urgent_cb_sessions_left", 0)
    if banned_side and sessions > 0:
        return True, banned_side
    return False, ""


def check_wti_slot(wti_pct: float) -> Tuple[bool, str]:
    if wti_pct < WTI_CORR_THRESHOLD:
        return True, ""  # WTI rendah, tidak terpengaruh limit

    current = count_correlated_positions()
    if current >= MAX_CORR_POSITIONS:
        reason = (
            f"WTI {wti_pct:.1f}% >= {WTI_CORR_THRESHOLD}% "
            f"dan sudah ada {current}/{MAX_CORR_POSITIONS} posisi highly-correlated"
        )
        return False, reason

    return True, ""


def save_daily_stats() -> Optional[str]:
    try:
        from order.executor import get_available_balance as _live_bal
        modal_basis = _live_bal()
        if modal_basis <= 0:
            modal_basis = 350.0
    except Exception:
        modal_basis = 350.0

    try:
        if not os.path.exists(_PAPER_HIST_FILE):
            return None
        with open(_PAPER_HIST_FILE) as f:
            history = json.load(f)

        today     = datetime.now(timezone.utc).date()
        today_str = today.isoformat()

        today_trades = []
        for h in history:
            closed_at = h.get("closed_at")
            if closed_at:
                closed_date = datetime.fromtimestamp(closed_at, tz=timezone.utc).date()
                if closed_date == today:
                    today_trades.append(h)

        if not today_trades:
            return None

        pnls    = [float(t.get("pnl", 0)) for t in today_trades]
        total   = sum(pnls)
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p <= 0]
        winrate = len(wins) / len(pnls) * 100 if pnls else 0

        by_symbol: Dict[str, List[float]] = {}
        for t in today_trades:
            sym = t.get("symbol", "?")
            by_symbol.setdefault(sym, []).append(float(t.get("pnl", 0)))

        symbol_summary = {
            sym: {
                "trades": len(p),
                "pnl":    round(sum(p), 4),
                "wins":   sum(1 for x in p if x > 0),
            }
            for sym, p in by_symbol.items()
        }

        stats = {
            "date":           today_str,
            "total_trades":   len(today_trades),
            "total_pnl":      round(total, 4),
            "pnl_pct":        round(total / modal_basis * 100, 2),
            "wins":           len(wins),
            "losses":         len(losses),
            "winrate_pct":    round(winrate, 1),
            "best_trade":     round(max(pnls), 4) if pnls else 0,
            "worst_trade":    round(min(pnls), 4) if pnls else 0,
            "by_symbol":      symbol_summary,
            "trades":         today_trades,
            "saved_at":       datetime.now(timezone.utc).isoformat(),
        }

        path = os.path.join(_HISTORY_DIR, f"{today_str}.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info("[risk] Daily stats saved: %s (trades=%d pnl=%.2f)",
                    path, len(today_trades), total)
        return path

    except Exception as e:
        logger.error("[risk] Gagal save daily stats: %s", e)
        return None


def get_daily_summary_text() -> str:
    """Return teks ringkasan hari ini untuk Telegram."""
    try:
        from order.executor import get_available_balance as _live_bal
        modal_basis = _live_bal()
        if modal_basis <= 0:
            modal_basis = 350.0
    except Exception:
        modal_basis = 350.0

    today = datetime.now(timezone.utc).date().isoformat()
    path  = os.path.join(_HISTORY_DIR, f"{today}.json")

    if os.path.exists(path):
        try:
            with open(path) as f:
                s = json.load(f)
        except Exception:
            s = None
    else:
        s = None

    if s is None:
        pnl     = _get_today_pnl()
        pnl_pct = pnl / modal_basis * 100
        return (
            f"📅 <b>Hari Ini ({today})</b>\n"
            f"  PnL : <b>{'+'if pnl>=0 else ''}{pnl:.2f} USDT ({pnl_pct:+.2f}%)</b>"
        )

    pnl_str = f"+{s['total_pnl']:.2f}" if s['total_pnl'] >= 0 else f"{s['total_pnl']:.2f}"
    emoji   = "📈" if s["total_pnl"] >= 0 else "📉"

    lines = [
        f"📅 <b>Ringkasan Hari Ini — {today}</b>",
        f"  {emoji} PnL       : <b>{pnl_str} USDT ({s['pnl_pct']:+.2f}%)</b>",
        f"  Trades    : {s['total_trades']}  ({s['wins']}W / {s['losses']}L  {s['winrate_pct']:.0f}% WR)",
        f"  Best      : <code>+{s['best_trade']:.2f}</code>",
        f"  Worst     : <code>{s['worst_trade']:.2f}</code>",
    ]

    if s.get("by_symbol"):
        lines.append("\n  Per simbol:")
        for sym, info in sorted(s["by_symbol"].items(), key=lambda x: -abs(x[1]["pnl"])):
            p = info["pnl"]
            lines.append(f"    {'+'if p>=0 else ''}{p:.2f} USDT  {sym} ({info['wins']}W/{info['trades']-info['wins']}L)")

    return "\n".join(lines)


def get_floating_drawdown() -> Tuple[float, float]:
    """
    Hitung floating PnL semua posisi aktif (unrealized).
    Return (total_floating_pnl, drawdown_pct_dari_modal).
    """
    import requests

    try:
        from order.executor import get_available_balance as _live_bal
        modal_basis = _live_bal()
        if modal_basis <= 0:
            modal_basis = 350.0
    except Exception:
        modal_basis = 350.0

    try:
        if not os.path.exists(_PAPER_POS_FILE):
            return 0.0, 0.0
        with open(_PAPER_POS_FILE) as f:
            positions = json.load(f)
        open_pos = [p for p in positions if p.get("status") == "open"]
        if not open_pos:
            return 0.0, 0.0

        total_float = 0.0
        for pos in open_pos:
            sym      = pos.get("symbol", "")
            entry    = float(pos.get("entry_price", 0))
            notional = float(pos.get("notional", 0))
            side     = pos.get("side", "BUY")
            if entry <= 0 or notional <= 0:
                continue
            try:
                resp = requests.get(
                    "https://fapi.binance.com/fapi/v1/premiumIndex",
                    params={"symbol": sym},
                    timeout=3,
                )
                mark = float(resp.json().get("markPrice", 0))
                if mark <= 0:
                    continue
                if side == "BUY":
                    pnl = (mark - entry) / entry * notional
                else:
                    pnl = (entry - mark) / entry * notional
                total_float += pnl
            except Exception:
                continue

        dd_pct = (-total_float) / modal_basis if total_float < 0 else 0.0
        return round(total_float, 4), round(dd_pct, 4)

    except Exception as e:
        logger.warning("[risk] get_floating_drawdown error: %s", e)
        return 0.0, 0.0

def get_ban_list() -> Dict[str, int]:
    state = _load_state()
    return dict(state.get("sl_bans", {}))


def get_risk_summary_text() -> str:
    """Teks lengkap status risk untuk /stats di Telegram."""
    state   = _load_state()
    bans    = state.get("sl_bans", {})
    cb_left = state.get("cb_sessions_left", 0)
    corr    = count_correlated_positions()
    today   = get_daily_summary_text()
    float_pnl, dd_pct = get_floating_drawdown()

    lines = [today, ""]

    # Floating PnL
    fp_str  = f"+{float_pnl:.2f}" if float_pnl >= 0 else f"{float_pnl:.2f}"
    fp_emoji = "📈" if float_pnl >= 0 else "📉"
    lines.append(f"{fp_emoji} Floating PnL : <b>{fp_str} USDT</b>")
    if dd_pct >= 0.10:
        lines.append(f"  ⚠️ Floating drawdown {dd_pct*100:.1f}% — waspada!")
    lines.append("")

    if cb_left > 0:
        lines.append(f"⚡ <b>Circuit Breaker AKTIF</b> — {cb_left} sesi lagi")
    else:
        today_pnl  = _get_today_pnl()
        try:
            from order.executor import get_available_balance as _lb
            _modal = _lb() or 350.0
        except Exception:
            _modal = 350.0
        loss_pct    = max(-today_pnl / _modal * 100, 0)
        cb_headroom = CIRCUIT_BREAKER_PCT * 100 - loss_pct
        lines.append(
            f"✅ Circuit Breaker : OFF  "
            f"(loss hari ini {loss_pct:.1f}%, headroom {cb_headroom:.1f}%)"
        )

    lines.append(
        f"🔗 Posisi WTI ≥{WTI_CORR_THRESHOLD:.0f}% : "
        f"<b>{corr}/{MAX_CORR_POSITIONS}</b>"
    )

    # SL bans
    if bans:
        ban_lines = ", ".join(f"{s}({r}sess)" for s, r in sorted(bans.items()))
        lines.append(f"🚫 SL Ban : {ban_lines}")
    else:
        lines.append("🚫 SL Ban : tidak ada")

    # Urgent CB
    ucb_ban, ucb_side = get_urgent_cb_ban()
    ucb_triggered, ucb_dir = is_urgent_cb_triggered()
    if ucb_triggered:
        if ucb_ban:
            lines.append(f"🚨 Urgent CB : BTC {ucb_dir} — <b>{ucb_side} di-ban 1 sesi</b>")
        else:
            lines.append(f"🚨 Urgent CB : BTC {ucb_dir} — ban sudah selesai")
    else:
        lines.append("🚨 Urgent CB : tidak aktif")

    return "\n".join(lines)
