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
)

logger = logging.getLogger(__name__)

MAX_OPEN_POSITIONS   = 10
MAX_ORDERS_PER_SCAN  = 2
FLOATING_DD_WARN_PCT = 0.10   # warning kalau floating loss >= 10%


def _count_active_positions() -> int:
    try:
        from config import PAPER_TRADING_MODE
        if PAPER_TRADING_MODE:
            from order.paper_executor import _load_positions
            positions = _load_positions()
            return sum(1 for p in positions if p.get("status") == "open")
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


async def _broadcast(bot: Bot, message: str) -> None:
    for chat_id in ALLOWED_CHAT_IDS:
        try:
            await bot.send_message(chat_id=chat_id, text=message, parse_mode=ParseMode.HTML)
        except Exception as e:
            err = str(e)
            # Forbidden = user belum /start bot — log debug saja, jangan spam WARNING
            if "Forbidden" in err or "bot can't initiate" in err:
                logger.debug("[scheduler] Chat %s belum /start bot — skip", chat_id)
            else:
                logger.warning("[scheduler] Gagal kirim ke %s: %s", chat_id, e)


def _make_notify(event_loop: asyncio.AbstractEventLoop, bot: Bot):
    def sync_notify(msg: str) -> None:
        try:
            if event_loop.is_closed():
                return
            future = asyncio.run_coroutine_threadsafe(_broadcast(bot, msg), event_loop)
            future.result(timeout=15)
        except Exception as e:
            logger.warning("[scheduler] notify error: %s", e)
    return sync_notify


async def run_session(bot: Bot) -> None:
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_id = datetime.now(timezone.utc).strftime("%H%M")
    loop       = asyncio.get_event_loop()

    logger.info("[scheduler] ===== SESSION %s START =====", session_id)

    # Tick risk counters (ban, CB)
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
        )
        return
    
    # ── BTC Volume Spike Cooldown ────────────────────────────────────────────
    spike_active, spike_remaining, spike_reason = is_btc_spike_cooldown()
    if spike_active:
        remaining_min = int(spike_remaining / 60)
        await _broadcast(
            bot,
            f"🌊 <b>BTC Spike CB — #{session_id} di-skip</b> — {now}\n"
            f"  Cooldown tersisa : <b>{remaining_min} menit</b>\n"
            f"  Sebab            : <i>{spike_reason}</i>\n"
            f"  Posisi berjalan  : dibiarkan aktif",
        )
        return

    # Posisi penuh
    active_count = _count_active_positions()
    if active_count >= MAX_OPEN_POSITIONS:
        await _broadcast(
            bot,
            f"⏸ <b>Sesi #{session_id} di-skip</b> — {now}\n"
            f"  Posisi aktif : <b>{active_count}/{MAX_OPEN_POSITIONS}</b> (penuh)",
        )
        return

    total_slots     = MAX_OPEN_POSITIONS - active_count
    slots_this_scan = min(total_slots, MAX_ORDERS_PER_SCAN)

    # Floating drawdown warning
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
        await _broadcast(bot, f"❌ <b>Scan error:</b> <code>{e}</code>")
        return

    await _broadcast(bot, format_scan_summary(passed, SCAN_TOP_N, DEFAULT_INTERVAL))

    if not passed:
        await _broadcast(bot, f"✅ <b>Sesi #{session_id} selesai</b> — tidak ada token lolos.")
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

         # Urgent CB ban check — side tertentu dilarang 1 sesi
        ucb_active, ucb_banned_side = get_urgent_cb_ban()
        if ucb_active:
            candidate_dir = candidate.get("direction", "")
            # candidate_dir = "LONG" atau "SHORT"
            if candidate_dir == ucb_banned_side:
                logger.info("[scheduler] %s skip — Urgent CB ban %s", symbol, ucb_banned_side)
                await _broadcast(
                    bot,
                    f"🚨 <b>{symbol}</b> skip — Urgent CB: <b>{ucb_banned_side} di-ban 1 sesi</b>"
                )
                continue

        # SL Ban check
        banned, ban_rem = is_banned(symbol)
        if banned:
            logger.info("[scheduler] %s banned %d sesi — skip", symbol, ban_rem)
            await _broadcast(bot, f"🚫 <b>{symbol}</b> — banned {ban_rem} sesi (kena SL)")
            continue

        # WTI check sebelum pipeline
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
            await _broadcast(bot, f"🔗 <b>{symbol}</b> skip WTI — <i>{wti_reason}</i>")
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
                await _broadcast(bot, f"❌ <b>Pipeline error {symbol}:</b> <code>{e}</code>")
            continue

    if loop.is_closed():
        return

    final_active = _count_active_positions()
    await _broadcast(
        bot,
        f"✅ <b>Sesi #{session_id} selesai</b>\n"
        f"  Token discan   : {SCAN_TOP_N}\n"
        f"  Token lolos    : {len(passed)}\n"
        f"  Pipeline dicoba: {pipeline_tried}\n"
        f"  Order placed   : <b>{orders_placed}/{slots_this_scan}</b>\n"
        f"  Posisi aktif   : <b>{final_active}/{MAX_OPEN_POSITIONS}</b>\n"
        f"  Next session   : {SCHEDULE_INTERVAL_MINUTES} menit lagi",
    )
    logger.info("[scheduler] SESSION %s END orders=%d active=%d", session_id, orders_placed, final_active)
    _try_save_daily(bot, loop)


def _try_save_daily(bot: Bot, loop) -> None:
    try:
        path = save_daily_stats()
        if path:
            summary = get_daily_summary_text()
            asyncio.run_coroutine_threadsafe(_broadcast(bot, summary), loop)
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
