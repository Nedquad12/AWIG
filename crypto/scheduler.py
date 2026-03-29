"""
scheduler.py — Automated Trading Scheduler

Jalankan tiap 30 menit:
  1. Scan top 100 token → filter by score threshold
  2. Per token yang lolos: jalankan full pipeline (ML + AI + order)
  3. Stop setelah 1 order berhasil di-place (MAX_ORDERS_PER_SESSION = 1)
  4. Kirim ringkasan sesi ke Telegram

Dipanggil dari main.py saat bot start.
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


# ------------------------------------------------------------------
# Helper: kirim pesan ke semua allowed chat ID
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Satu sesi scan + pipeline
# ------------------------------------------------------------------

async def run_session(bot: Bot) -> None:
    """Jalankan satu sesi lengkap: scan → pipeline → max 1 order."""
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_id = datetime.now(timezone.utc).strftime("%H%M")

    logger.info("[scheduler] ===== SESSION %s START =====", session_id)
    await _broadcast(bot, f"⏰ <b>Sesi Otomatis #{session_id}</b> — {now}\n🔍 Memulai scan {SCAN_TOP_N} token...")

    # ── Tahap 1: Scan ────────────────────────────────────────────
    try:
        passed = await asyncio.get_event_loop().run_in_executor(
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

    # Kirim ringkasan scan
    scan_msg = format_scan_summary(passed, SCAN_TOP_N, DEFAULT_INTERVAL)
    await _broadcast(bot, scan_msg)

    if not passed:
        await _broadcast(bot, f"✅ <b>Sesi #{session_id} selesai</b> — tidak ada order.")
        return

    # ── Tahap 2: Pipeline per token, stop setelah 1 order ────────
    orders_placed = 0
    pipeline_tried = 0

    for candidate in passed:
        if orders_placed >= MAX_ORDERS_PER_SESSION:
            break

        symbol = candidate["symbol"]
        pipeline_tried += 1

        logger.info("[scheduler] Running pipeline for %s...", symbol)

        # notify callback
        async def _notify(msg: str, _bot=bot):
            await _broadcast(_bot, msg)

        def sync_notify(msg: str):
            asyncio.run_coroutine_threadsafe(
                _broadcast(bot, msg),
                asyncio.get_event_loop(),
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=symbol: run_pipeline(
                    s,
                    interval=DEFAULT_INTERVAL,
                    notify=sync_notify,
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


# ------------------------------------------------------------------
# Public: setup_scheduler
# ------------------------------------------------------------------

def setup_scheduler(bot: Bot) -> AsyncIOScheduler:
    """
    Buat dan return AsyncIOScheduler yang sudah dikonfigurasi.
    Panggil scheduler.start() setelah ini.

    Sesi pertama langsung jalan saat start (next_run_time=now),
    lalu tiap SCHEDULE_INTERVAL_MINUTES menit.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        func=run_session,
        trigger=IntervalTrigger(minutes=SCHEDULE_INTERVAL_MINUTES),
        args=[bot],
        id="auto_trading_session",
        name=f"Auto Trading ({SCHEDULE_INTERVAL_MINUTES}m)",
        replace_existing=True,
        next_run_time=datetime.now(timezone.utc),   # langsung run saat start
    )

    logger.info(
        "[scheduler] Scheduler configured: every %d minutes, first run immediately.",
        SCHEDULE_INTERVAL_MINUTES,
    )
    return scheduler
