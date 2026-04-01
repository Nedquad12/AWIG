"""
scheduler.py — Automated Trading Scheduler

Setiap sesi dimulai dengan membersihkan folder weights agar bobot
tidak basi dari sesi sebelumnya. Weights di-generate fresh per sesi.
"""

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
from scanner  import scan, format_scan_summary
from pipeline import run as run_pipeline

logger = logging.getLogger(__name__)

MAX_ORDERS_PER_SESSION = 1


# ------------------------------------------------------------------
# Bersihkan folder weights
# ------------------------------------------------------------------

def _clear_weights() -> int:
    """
    Hapus semua file .json di WEIGHTS_DIR sebelum sesi dimulai.
    Return jumlah file yang dihapus.
    """
    if not os.path.isdir(WEIGHTS_DIR):
        logger.info("[scheduler] Folder weights belum ada, skip clear.")
        return 0

    files   = glob.glob(os.path.join(WEIGHTS_DIR, "*.json"))
    deleted = 0
    for f in files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            logger.warning("[scheduler] Gagal hapus %s: %s", f, e)

    logger.info("[scheduler] Cleared %d weight file(s) dari %s", deleted, WEIGHTS_DIR)
    return deleted


# ------------------------------------------------------------------
# Broadcast
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
            logger.warning(
                "[scheduler] Gagal kirim ke chat_id=%s: %s — "
                "pastikan user sudah /start bot.",
                chat_id, e,
            )


# ------------------------------------------------------------------
# Notify function aman untuk thread executor
# ------------------------------------------------------------------

def _make_notify(event_loop: asyncio.AbstractEventLoop, bot: Bot):
    def sync_notify(msg: str) -> None:
        try:
            if event_loop.is_closed():
                logger.debug("[scheduler] notify skip — event loop closed")
                return
            future = asyncio.run_coroutine_threadsafe(
                _broadcast(bot, msg), event_loop)
            future.result(timeout=15)
        except asyncio.CancelledError:
            logger.debug("[scheduler] notify cancelled (shutdown)")
        except RuntimeError as e:
            logger.debug("[scheduler] notify runtime error (likely shutdown): %s", e)
        except Exception as e:
            logger.warning("[scheduler] notify error: %s", e)
    return sync_notify


# ------------------------------------------------------------------
# Main session
# ------------------------------------------------------------------

async def run_session(bot: Bot) -> None:
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_id = datetime.now(timezone.utc).strftime("%H%M")
    loop       = asyncio.get_event_loop()

    logger.info("[scheduler] ===== SESSION %s START =====", session_id)

    # ── Bersihkan weights sebelum scan dimulai ────────────────────
    deleted = _clear_weights()

    await _broadcast(
        bot,
        f"⏰ <b>Sesi Otomatis #{session_id}</b> — {now}\n"
        f"🗑 Cleared {deleted} weight file(s)\n"
        f"🔍 Memulai scan {SCAN_TOP_N} token...",
    )

    # ── Scan ─────────────────────────────────────────────────────
    try:
        passed = await loop.run_in_executor(
            None,
            lambda: scan(
                top_n=SCAN_TOP_N,
                interval=DEFAULT_INTERVAL,
                threshold=SCAN_SCORE_THRESHOLD,
            ),
        )
    except GeneratorExit:
        logger.info("[scheduler] Session %s interrupted (scan)", session_id)
        return
    except Exception as e:
        logger.exception("[scheduler] Scan error: %s", e)
        await _broadcast(bot, f"❌ <b>Scan error:</b> <code>{e}</code>")
        return

    await _broadcast(bot, format_scan_summary(passed, SCAN_TOP_N, DEFAULT_INTERVAL))

    if not passed:
        await _broadcast(bot, f"✅ <b>Sesi #{session_id} selesai</b> — tidak ada token lolos scan.")
        return

    # ── Pipeline per token ────────────────────────────────────────
    orders_placed  = 0
    pipeline_tried = 0

    for candidate in passed:
        if orders_placed >= MAX_ORDERS_PER_SESSION:
            break

        if loop.is_closed():
            logger.info("[scheduler] Loop closed — stop session %s", session_id)
            return

        symbol         = candidate["symbol"]
        pipeline_tried += 1

        logger.info("[scheduler] Running pipeline for %s...", symbol)

        try:
            result = await loop.run_in_executor(
                None,
                lambda s=symbol: run_pipeline(
                    s,
                    interval=DEFAULT_INTERVAL,
                    notify=_make_notify(loop, bot),
                ),
            )

            if result.get("stage") == "completed" and result.get("order", {}).get("ok"):
                orders_placed += 1
                logger.info("[scheduler] Order placed for %s. Session limit reached.", symbol)
                break

        except GeneratorExit:
            logger.info("[scheduler] Session %s interrupted during pipeline for %s", session_id, symbol)
            return
        except asyncio.CancelledError:
            logger.info("[scheduler] Session %s cancelled", session_id)
            return
        except Exception as e:
            logger.exception("[scheduler] Pipeline error for %s: %s", symbol, e)
            if not loop.is_closed():
                await _broadcast(bot, f"❌ <b>Pipeline error {symbol}:</b> <code>{e}</code>")
            continue

    # ── Ringkasan sesi ────────────────────────────────────────────
    if loop.is_closed():
        return

    await _broadcast(
        bot,
        f"✅ <b>Sesi #{session_id} selesai</b>\n"
        f"  Token discan    : {SCAN_TOP_N}\n"
        f"  Token lolos     : {len(passed)}\n"
        f"  Pipeline dicoba : {pipeline_tried}\n"
        f"  Order placed    : <b>{orders_placed}</b>\n"
        f"  Next session    : {SCHEDULE_INTERVAL_MINUTES} menit lagi",
    )
    logger.info("[scheduler] ===== SESSION %s END — orders=%d =====", session_id, orders_placed)


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

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
        "[scheduler] Configured: every %d minutes, first run immediately.",
        SCHEDULE_INTERVAL_MINUTES,
    )
    return scheduler
