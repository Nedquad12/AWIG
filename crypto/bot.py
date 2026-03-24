"""
bot.py — Telegram Bot untuk Crypto Futures Trading

Commands:
  /scan SYMBOL      — Full pipeline: fetch → skor → ML → AI → trade + SL/TP
  /dryrun SYMBOL    — Full pipeline tanpa eksekusi order
  /skor SYMBOL      — Hanya hitung skor (cepat, tanpa ML/AI)
  /scanall          — Scan semua pair dari config
  /dryrunall        — Dry run semua pair dari config
  /pos              — Lihat posisi terbuka + info SL/TP
  /close SYMBOL     — Tutup posisi (cancel SL/TP dulu, lalu market close)
  /balance          — Cek balance USDT Futures Testnet
  /pairs            — Lihat pair yang dikonfigurasi
  /help             — Bantuan lengkap
"""

import asyncio
import logging
import os
import sys

from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_TOPIC_ID,
    TRADE_PAIRS, CACHE_DIR, WEIGHTS_DIR, HISTORY_DIR, LOG_DIR,
)
from pipeline import run_full_pipeline, run_scores_only
from api_binance import get_account_balance, get_open_positions
from trader import close_trade, load_positions
from formatter import (
    fmt_pipeline_result, fmt_scores_only,
    fmt_positions, fmt_close_result, fmt_help,
)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level    = logging.INFO,
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Pastikan direktori ada
for d in (CACHE_DIR, WEIGHTS_DIR, HISTORY_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _thread_kwargs() -> dict:
    return {"message_thread_id": TELEGRAM_TOPIC_ID} if TELEGRAM_TOPIC_ID else {}


async def _reply(update: Update, text: str):
    await update.message.reply_text(
        text,
        parse_mode               = ParseMode.HTML,
        disable_web_page_preview = True,
        **_thread_kwargs(),
    )


async def _reply_long(update: Update, text: str):
    """Kirim pesan panjang dengan auto-split per 4000 karakter."""
    if len(text) <= 4000:
        await _reply(update, text)
        return
    chunks = [text[i:i+3900] for i in range(0, len(text), 3900)]
    for chunk in chunks:
        await _reply(update, chunk)
        await asyncio.sleep(0.4)


def _sym(args: list) -> str:
    return args[0].upper() if args else ""


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply(update, fmt_help())


async def cmd_pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["⚙️ <b>Pair yang dikonfigurasi:</b>", ""]
    for p in TRADE_PAIRS:
        lines.append(f"  • <code>{p}</code>")
    lines.append(f"\nTotal: {len(TRADE_PAIRS)} pair")
    await _reply(update, "\n".join(lines))


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _reply(update, "⏳ Mengambil balance...")
    try:
        bal = get_account_balance("USDT")
        await _reply(update, f"💰 <b>Balance Futures Testnet</b>\n\nUSDT tersedia: <b>{bal:.4f}</b>")
    except Exception as e:
        await _reply(update, f"❌ Gagal ambil balance: {e}")


async def cmd_pos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tampilkan posisi terbuka dari Binance + SL/TP info dari local state."""
    await _reply(update, "⏳ Mengambil posisi terbuka...")
    try:
        binance_pos  = get_open_positions()
        local_state  = load_positions()
        await _reply(update, fmt_positions(binance_pos, local_state))
    except Exception as e:
        await _reply(update, f"❌ Error: {e}")


async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Tutup posisi untuk symbol tertentu.
    Bot akan:
      1. Cancel semua open order SL/TP terlebih dahulu
      2. Place market order untuk flat posisi
    """
    symbol = _sym(context.args)
    if not symbol:
        await _reply(update, "⚠️ Format: <code>/close BTCUSDT</code>")
        return

    await _reply(
        update,
        f"⏳ Menutup posisi <b>{symbol}</b>...\n"
        f"(cancel SL/TP dulu, lalu market close)"
    )

    result = close_trade(symbol, reason="manual via /close")
    await _reply(update, fmt_close_result(result))


async def cmd_skor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = _sym(context.args)
    if not symbol:
        await _reply(update, "⚠️ Format: <code>/skor BTCUSDT</code>")
        return
    await _reply(update, f"⏳ Menghitung skor <b>{symbol}</b>...")
    result = run_scores_only(symbol)
    await _reply(update, fmt_scores_only(result))


async def _run_scan(update: Update, symbol: str, execute: bool):
    """Jalankan full pipeline dan kirim hasilnya."""
    mode = "LIVE 🔴" if execute else "DRY RUN 🟡"
    await _reply(
        update,
        f"🚀 <b>{symbol}</b> — Memulai [{mode}]\n"
        f"<i>fetch → skor → history → ML adj → ML pred → AI → "
        f"{'entry + SL/TP' if execute else 'skip order'}</i>"
    )
    try:
        result = run_full_pipeline(symbol, execute=execute)
        await _reply_long(update, fmt_pipeline_result(result))
    except Exception as e:
        logger.error(f"Pipeline error {symbol}: {e}", exc_info=True)
        await _reply(update, f"❌ <b>{symbol}</b> — Pipeline exception:\n<code>{e}</code>")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = _sym(context.args)
    if not symbol:
        await _reply(update, "⚠️ Format: <code>/scan BTCUSDT</code>")
        return
    await _run_scan(update, symbol, execute=True)


async def cmd_dryrun(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = _sym(context.args)
    if not symbol:
        await _reply(update, "⚠️ Format: <code>/dryrun BTCUSDT</code>")
        return
    await _run_scan(update, symbol, execute=False)


async def cmd_scanall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs = TRADE_PAIRS
    if not pairs:
        await _reply(update, "⚠️ Tidak ada pair di TRADE_PAIRS (config.py)")
        return
    await _reply(update, f"🔁 <b>Scan semua {len(pairs)} pair [LIVE]</b>\n" + "  ".join(pairs))
    for symbol in pairs:
        await _run_scan(update, symbol, execute=True)
        await asyncio.sleep(3)
    await _reply(update, f"✅ Selesai scan semua {len(pairs)} pair.")


async def cmd_dryrunall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs = TRADE_PAIRS
    if not pairs:
        await _reply(update, "⚠️ Tidak ada pair di TRADE_PAIRS (config.py)")
        return
    await _reply(update, f"🔁 <b>Dry run semua {len(pairs)} pair</b>\n" + "  ".join(pairs))
    for symbol in pairs:
        await _run_scan(update, symbol, execute=False)
        await asyncio.sleep(3)
    await _reply(update, f"✅ Selesai dry run semua {len(pairs)} pair.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("Starting Crypto Trading Bot (Binance Futures Testnet)...")

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(60)
        .write_timeout(60)
        .pool_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("help",       cmd_help))
    app.add_handler(CommandHandler("pairs",      cmd_pairs))
    app.add_handler(CommandHandler("balance",    cmd_balance))
    app.add_handler(CommandHandler("pos",        cmd_pos))
    app.add_handler(CommandHandler("close",      cmd_close))
    app.add_handler(CommandHandler("skor",       cmd_skor))
    app.add_handler(CommandHandler("scan",       cmd_scan))
    app.add_handler(CommandHandler("dryrun",     cmd_dryrun))
    app.add_handler(CommandHandler("scanall",    cmd_scanall))
    app.add_handler(CommandHandler("dryrunall",  cmd_dryrunall))

    logger.info("Bot ready. Listening for Telegram commands...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
