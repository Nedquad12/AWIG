"""
scheduler.py — Automated Trading Scheduler
"""

import asyncio
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
    TELEGRAM_BOT_TOKEN,
)
from scanner  import scan, format_scan_summary
from pipeline import run as run_pipeline

logger = logging.getLogger(__name__)

MAX_ORDERS_PER_SESSION = 1


async def _broadcast(bot: Bot, message: str) -> None:
    for chat_id in ALLOWED_CHAT_IDS:
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.warning("[scheduler] broadcast error chat_id=%s: %s", chat_id, e)


async def run_session(bot: Bot) -> None:
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_id = datetime.now(timezone.utc).strftime("%H%M")
    loop       = asyncio.get_event_loop()

    logger.info("[scheduler] ===== SESSION %s START =====", session_id)
    await _broadcast(bot, f"⏰ <b>Sesi Otomatis #{session_id}</b> — {now}\n🔍 Memulai scan {SCAN_TOP_N} token...")

    # ── Tahap 1: Scan ────────────────────────────────────────────
    try:
        passed = await loop.run_in_executor(
            None,
            lambda: scan(
                top_n=SCAN_TOP_N,
                interval=DEFAULT_INTERVAL,
                threshold=SCAN_SCORE_THRESHOLD,
            )
        )
    except Exception as e:
        logger.exception("[scheduler] Scan error: %s", e)
        await _broadcast(bot, f"❌ <b>Scan error:</b> <code>{e}</code>")
        return

    scan_msg = format_scan_summary(passed, SCAN_TOP_N, DEFAULT_INTERVAL)
    await _broadcast(bot, scan_msg)

    if not passed:
        await _broadcast(bot, f"✅ <b>Sesi #{session_id} selesai</b> — tidak ada order.")
        return

    # ── Tahap 2: Pipeline per token ───────────────────────────────
    orders_placed  = 0
    pipeline_tried = 0

    for candidate in passed:
        if orders_placed >= MAX_ORDERS_PER_SESSION:
            break

        symbol = candidate["symbol"]
        pipeline_tried += 1
        logger.info("[scheduler] Running pipeline for %s...", symbol)

        # Buat notify function yang thread-safe
        # Pipeline jalan di executor (thread), broadcast harus dijadwalkan ke loop utama
        def make_notify(event_loop, telegram_bot):
            def sync_notify(msg: str):
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        _broadcast(telegram_bot, msg),
                        event_loop,
                    )
                    future.result(timeout=10)  # tunggu sampai terkirim
                except Exception as ex:
                    logger.warning("[scheduler] notify error: %s", ex)
            return sync_notify

        notify_fn = make_notify(loop, bot)

        try:
            result = await loop.run_in_executor(
                None,
                lambda s=symbol: run_pipeline(
                    s,
                    interval=DEFAULT_INTERVAL,
                    notify=notify_fn,
                )
            )

            stage = result.get("stage", "")
            if stage == "completed" and result.get("order", {}).get("ok"):
                orders_placed += 1
                logger.info("[scheduler] Order placed for %s. Session limit reached.", symbol)
                break

        except Exception as e:
            logger.exception("[scheduler] Pipeline error for %s: %s", symbol, e)
            await _broadcast(bot, f"❌ <b>Pipeline error {symbol}:</b> <code>{e}</code>")
            continue

    # ── Ringkasan sesi ────────────────────────────────────────────
    summary = (
        f"✅ <b>Sesi #{session_id} selesai</b>\n"
        f"  Token discan    : {SCAN_TOP_N}\n"
        f"  Token lolos     : {len(passed)}\n"
        f"  Pipeline dicoba : {pipeline_tried}\n"
        f"  Order placed    : <b>{orders_placed}</b>\n"
        f"  Next session    : {SCHEDULE_INTERVAL_MINUTES} menit lagi"
    )
    await _broadcast(bot, summary)
    logger.info("[scheduler] ===== SESSION %s END — orders=%d =====", session_id, orders_placed)


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

    logger.info(
        "[scheduler] Scheduler configured: every %d minutes, first run immediately.",
        SCHEDULE_INTERVAL_MINUTES,
    )
    return scheduler
