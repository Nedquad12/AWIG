import asyncio
import glob
import logging
import os
import sys
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval  import IntervalTrigger
from telegram import Bot
from telegram.constants import ParseMode

sys.path.append(os.path.dirname(__file__))
from config import (
    ALLOWED_CHAT_IDS,
    DEFAULT_INTERVAL,
    SCAN_SCORE_THRESHOLD,
    SCAN_TOP_N,
    SCHEDULE_INTERVAL_MINUTES,
    WEIGHTS_DIR,
)
from scanner      import scan, format_scan_summary
from pipeline     import run as run_pipeline
from risk_manager import (
    tick_session,
    check_circuit_breaker,
    check_wti_slot,
    is_banned,
    get_urgent_cb_ban,
    save_daily_stats,
    get_daily_summary_text,
    get_floating_drawdown,
    is_btc_spike_cooldown,
)

logger = logging.getLogger(__name__)

MAX_OPEN_POSITIONS   = 10
MAX_ORDERS_PER_SCAN  = 4   # max order dibuka per sesi
MAX_FILLED_PER_SESSION = 2  # max order filled per sesi sebelum sisa di-cancel
FLOATING_DD_WARN_PCT = 0.10

# ── Telegram Topic Config ─────────────────────────────────────────────────────
GROUP_ID      = -1003758450134
TOPIC_BOARD   = 8    # output board.py / pipeline verdict
TOPIC_CB      = 6    # circuit breaker (daily loss, BTC spike, urgent CB)
TOPIC_ORDERS  = 2    # pending, filled, SL, TP, partial TP, trailing, breakeven, expired
TOPIC_ERROR   = 10   # error
TOPIC_GENERAL = 88   # semua yang tidak disebutkan


def _topic_for(msg: str) -> int:
    """Auto-detect topic berdasarkan isi pesan."""

    # CB — harus dicek pertama karena paling kritis
    cb_keywords = [
        "Circuit Breaker", "BTC Spike CB", "URGENT Circuit Breaker", "Urgent CB",
    ]
    if any(k in msg for k in cb_keywords):
        return TOPIC_CB

    # Orders — event posisi
    order_keywords = [
        "PAPER LIMIT", "LIMIT FILLED", "LIMIT CANCELLED", "Order Expired",
        "CLOSED", "Breakeven", "Trailing SL", "Partial TP",
        "Monitor aktif —", "Monitor Bot Online",
    ]
    if any(k in msg for k in order_keywords):
        return TOPIC_ORDERS

    # Board — verdict akhir pipeline (BUYING/SELLING/SKIP)
    board_keywords = [
        "🔲", "Board", "BUYING", "SELLING",
        "✅ SKIP", "⏭️ SKIP", "board verdict",
    ]
    if any(k in msg for k in board_keywords):
        return TOPIC_BOARD

    # General — hanya sesi mulai dan scan result yang boleh ke topic 88
    general_keywords = ["Sesi #", "Scan Result"]
    if any(k in msg for k in general_keywords):
        return TOPIC_GENERAL

    # Pipeline output selain board → drop (return None)
    drop_keywords = [
        "Training ML", "Walk-Forward", "Fold Detail", "Regime",
        "ML Prediction", "WFV", "BTC Context", "⏳", "🌡️", "🔮",
        "skip WTI", "banned", "di-skip", "token lolos",
        "Pipeline", "Cleared",
    ]
    if any(k in msg for k in drop_keywords):
        return None  # tidak dikirim ke mana-mana

    # Error — dicek SETELAH pipeline agar ❌ di fold tidak nyangkut ke sini
    error_keywords = ["error", "Error", "gagal", "Gagal", "GAGAL"]
    if any(k in msg for k in error_keywords):
        return TOPIC_ERROR

    return TOPIC_GENERAL


async def _send_to_topic(bot: Bot, message: str, topic_id: int) -> None:
    """
    Kirim pesan ke topik tertentu di supergroup.
    Delay 1 detik antar pesan agar tidak kena flood control Telegram.
    topic_id=None → drop pesan (tidak dikirim ke mana-mana).
    """
    if topic_id is None:
        return  # drop — pipeline output yang tidak perlu dikirim
    await asyncio.sleep(1)  # jeda 1 detik, cukup untuk hindari flood (max ~20 msg/menit)
    try:
        await bot.send_message(
            chat_id           = GROUP_ID,
            text              = message,
            parse_mode        = ParseMode.HTML,
            message_thread_id = topic_id,
        )
    except Exception as e:
        err = str(e)
        if "Forbidden" in err or "bot can't initiate" in err:
            logger.debug("[scheduler] Bot belum di-invite ke grup — skip")
        else:
            logger.warning("[scheduler] Gagal kirim ke topic %d: %s", topic_id, e)


async def _broadcast(bot: Bot, message: str, topic_id: int = None) -> None:
    """Kirim ke supergroup. topic_id opsional — jika None, auto-detect."""
    tid = topic_id if topic_id is not None else _topic_for(message)
    await _send_to_topic(bot, message, tid)


def _make_notify(event_loop: asyncio.AbstractEventLoop, bot: Bot, topic_id: int = None):
    """
    Buat sync notify function untuk thread lain (pipeline, monitor).
    Fire-and-forget — pipeline tidak pernah di-block oleh notif Telegram.
    topic_id opsional — jika None, auto-detect per pesan.
    """
    def sync_notify(msg: str) -> None:
        try:
            if event_loop.is_closed():
                return
            tid = topic_id if topic_id is not None else _topic_for(msg)
            # Tidak .result() — biar pipeline tidak nunggu notif
            asyncio.run_coroutine_threadsafe(
                _send_to_topic(bot, msg, tid), event_loop
            )
        except Exception as e:
            logger.warning("[scheduler] notify error: %s", e)
    return sync_notify


def _count_active_positions() -> int:
    try:
        from config import PAPER_TRADING_MODE
        if PAPER_TRADING_MODE:
            from order.paper_executor import _load_positions
            positions = _load_positions()
            return sum(1 for p in positions if p.get("status") in ("open", "pending"))
        else:
            import hashlib, hmac, time, urllib.parse, requests
            from config import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TRADE_URL, RECV_WINDOW
            params = {"timestamp": int(time.time() * 1000), "recvWindow": RECV_WINDOW}
            qs = urllib.parse.urlencode(params)
            params["signature"] = hmac.new(
                BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
            resp = requests.get(
                f"{BINANCE_TRADE_URL}/fapi/v2/positionRisk",
                params=params,
                headers={"X-MBX-APIKEY": BINANCE_API_KEY},
                timeout=10,
            )
            resp.raise_for_status()
            return sum(1 for p in resp.json() if float(p.get("positionAmt", 0)) != 0)
    except Exception as e:
        logger.warning("[scheduler] Gagal hitung posisi: %s — anggap 0", e)
        return 0


def _clear_weights() -> int:
    if not os.path.isdir(WEIGHTS_DIR):
        return 0
    files   = glob.glob(os.path.join(WEIGHTS_DIR, "*.json"))
    deleted = 0
    for f in files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            logger.warning("[scheduler] Gagal hapus %s: %s", f, e)
    logger.info("[scheduler] Cleared %d weight file(s)", deleted)
    return deleted


async def run_session(bot: Bot) -> None:
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    loop       = asyncio.get_event_loop()

    logger.info("[scheduler] ===== SESSION %s START =====", session_id)

    risk_state = tick_session()

    # Circuit Breaker
    cb_active, loss_pct = check_circuit_breaker()
    if cb_active:
        cb_left = risk_state.get("cb_sessions_left", 0)
        await _broadcast(
            bot,
            f"⚡ <b>Circuit Breaker AKTIF — #{session_id} di-skip</b> — {now}\n"
            f"  Loss hari ini  : <b>{loss_pct*100:.1f}%</b> (limit 5%)\n"
            f"  Sesi diblokir  : <b>{cb_left}</b> tersisa\n"
            f"  Posisi berjalan: dibiarkan aktif",
            topic_id=TOPIC_CB,
        )
        return

    # BTC Volume Spike Cooldown
    spike_active, spike_remaining, spike_reason = is_btc_spike_cooldown()
    if spike_active:
        remaining_min = int(spike_remaining / 60)
        await _broadcast(
            bot,
            f"🌊 <b>BTC Spike CB — #{session_id} di-skip</b> — {now}\n"
            f"  Cooldown tersisa : <b>{remaining_min} menit</b>\n"
            f"  Sebab            : <i>{spike_reason}</i>\n"
            f"  Posisi berjalan  : dibiarkan aktif",
            topic_id=TOPIC_CB,
        )
        return

    # Posisi penuh
    active_count = _count_active_positions()
    if active_count >= MAX_OPEN_POSITIONS:
        logger.info("[scheduler] Sesi %s di-skip — posisi penuh %d/%d", session_id, active_count, MAX_OPEN_POSITIONS)
        return

    total_slots     = MAX_OPEN_POSITIONS - active_count
    slots_this_scan = min(total_slots, MAX_ORDERS_PER_SCAN)

    float_pnl, dd_pct = get_floating_drawdown()
    dd_warn = ""
    if dd_pct >= FLOATING_DD_WARN_PCT:
        dd_warn = (
            f"\n⚠️ Floating drawdown <b>{dd_pct*100:.1f}%</b> "
            f"({float_pnl:.2f} USDT unrealized)"
        )

    deleted = _clear_weights()

    await _broadcast(
        bot,
        f"⏰ <b>Sesi #{session_id}</b> — {now}\n"
        f"  Posisi aktif  : <b>{active_count}/{MAX_OPEN_POSITIONS}</b>\n"
        f"  Slot sesi ini : <b>{slots_this_scan}</b>\n"
        f"🗑 Cleared {deleted} weight(s){dd_warn}\n"
        f"🔍 Scan {SCAN_TOP_N} token...",
        topic_id=TOPIC_GENERAL,
    )

    try:
        passed = await loop.run_in_executor(
            None,
            lambda: scan(top_n=SCAN_TOP_N, interval=DEFAULT_INTERVAL, threshold=SCAN_SCORE_THRESHOLD),
        )
    except GeneratorExit:
        return
    except Exception as e:
        logger.exception("[scheduler] Scan error: %s", e)
        await _broadcast(bot, f"❌ <b>Scan error:</b> <code>{e}</code>", topic_id=TOPIC_ERROR)
        return

    await _broadcast(bot, format_scan_summary(passed, SCAN_TOP_N, DEFAULT_INTERVAL), topic_id=TOPIC_GENERAL)

    if not passed:
        logger.info("[scheduler] Sesi %s — tidak ada token lolos", session_id)
        _try_save_daily(bot, loop)
        return

    orders_placed  = 0
    pipeline_tried = 0

    for candidate in passed:

        if orders_placed >= slots_this_scan:
            logger.info("[scheduler] Slot per-scan penuh %d/%d", orders_placed, slots_this_scan)
            break

        active_now = _count_active_positions()
        if active_now >= MAX_OPEN_POSITIONS:
            logger.info("[scheduler] Total posisi penuh %d/%d", active_now, MAX_OPEN_POSITIONS)
            break

        if loop.is_closed():
            return

        symbol = candidate["symbol"]

        # Urgent CB ban check
        ucb_active, ucb_banned_side = get_urgent_cb_ban()
        if ucb_active:
            candidate_dir = candidate.get("direction", "")
            if candidate_dir == ucb_banned_side:
                logger.info("[scheduler] %s skip — Urgent CB ban %s", symbol, ucb_banned_side)
                await _broadcast(
                    bot,
                    f"🚨 <b>{symbol}</b> skip — Urgent CB: <b>{ucb_banned_side} di-ban 1 sesi</b>",
                    topic_id=TOPIC_CB,
                )
                continue

        # SL Ban check
        banned, ban_rem = is_banned(symbol)
        if banned:
            logger.info("[scheduler] %s banned %d sesi — skip", symbol, ban_rem)
            continue

        # WTI check
        wti_pct = 0.0
        try:
            from wti_crypto import get_wti
            wti_result = await loop.run_in_executor(None, get_wti, symbol)
            if wti_result:
                wti_pct = float(wti_result.get("wti_pct", 0.0))
        except Exception as e:
            logger.warning("[scheduler] WTI %s error: %s", symbol, e)

        wti_ok, wti_reason = check_wti_slot(wti_pct)
        if not wti_ok:
            logger.info("[scheduler] %s WTI filter: %s", symbol, wti_reason)
            continue

        pipeline_tried += 1
        logger.info("[scheduler] Pipeline %s (wti=%.1f%% slot=%d/%d)",
                    symbol, wti_pct, orders_placed, slots_this_scan)

        try:
            result = await loop.run_in_executor(
                None,
                lambda s=symbol, w=wti_pct: run_pipeline(
                    s,
                    interval=DEFAULT_INTERVAL,
                    notify=_make_notify(loop, bot),
                    wti_pct=w,
                    session_id=session_id,
                ),
            )

            if result.get("stage") == "completed" and result.get("order", {}).get("ok"):
                orders_placed += 1

        except GeneratorExit:
            return
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("[scheduler] Pipeline error %s: %s", symbol, e)
            if not loop.is_closed():
                await _broadcast(
                    bot,
                    f"❌ <b>Pipeline error {symbol}:</b> <code>{e}</code>",
                    topic_id=TOPIC_ERROR,
                )
            continue

    if loop.is_closed():
        return

    final_active = _count_active_positions()
    logger.info("[scheduler] SESSION %s END orders=%d active=%d", session_id, orders_placed, final_active)
    _try_save_daily(bot, loop)


def _try_save_daily(bot: Bot, loop) -> None:
    try:
        path = save_daily_stats()
        if path:
            summary = get_daily_summary_text()
            asyncio.run_coroutine_threadsafe(
                _send_to_topic(bot, summary, TOPIC_GENERAL), loop
            )
    except Exception as e:
        logger.warning("[scheduler] Daily stats error: %s", e)


def setup_scheduler(bot: Bot) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        func=run_session,
        trigger=IntervalTrigger(minutes=SCHEDULE_INTERVAL_MINUTES),
        args=[bot],
        id="auto_trading_session",
        name=f"Auto Trading ({SCHEDULE_INTERVAL_MINUTES}m)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc),
    )
    logger.info("[scheduler] Configured every %d minutes", SCHEDULE_INTERVAL_MINUTES)
    return scheduler
