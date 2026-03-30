# =============================================================
# telegram_bot.py — Telegram bot handler
# =============================================================

import asyncio
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

import binance_client as bc
from config import ALLOWED_CHAT_IDS, TELEGRAM_BOT_TOKEN

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Decorator: whitelist chat ID
# ------------------------------------------------------------------
def restricted(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if chat_id not in ALLOWED_CHAT_IDS:
            await update.message.reply_text("⛔ Akses ditolak.")
            logger.warning("Akses ditolak untuk chat_id=%s", chat_id)
            return
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


def _fmt(val, decimals=4) -> str:
    try:
        return f"{float(val):,.{decimals}f}"
    except Exception:
        return str(val)


# ------------------------------------------------------------------
# /start & /help
# ------------------------------------------------------------------
@restricted
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 <b>Binance Futures Demo Bot</b>\n\n"
        "📊 <b>Akun &amp; Saldo</b>\n"
        "  /saldo — Cek saldo akun\n"
        "  /akun — Info akun lengkap\n\n"
        "📈 <b>Posisi</b>\n"
        "  /posisi — Semua posisi aktif\n"
        "  /posisi BTCUSDT — Posisi simbol tertentu\n\n"
        "📋 <b>Order</b>\n"
        "  /order — Semua open order\n"
        "  /riwayat BTCUSDT — 10 order terakhir\n\n"
        "💰 <b>Harga</b>\n"
        "  /harga BTCUSDT — Harga terakhir\n"
        "  /24jam BTCUSDT — Statistik 24 jam\n\n"
        "🔍 <b>Scan</b>\n"
        "  /scan — Lihat skor semua top 100 token\n\n"
        "🔧 <b>Lainnya</b>\n"
        "  /ping — Cek koneksi ke Binance\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

cmd_help = cmd_start


# ------------------------------------------------------------------
# /ping
# ------------------------------------------------------------------
@restricted
async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        import time
        server_ms = bc.get_server_time()
        local_ms  = int(time.time() * 1000)
        diff      = local_ms - server_ms
        await update.message.reply_text(
            f"✅ Terhubung ke Binance Testnet\n"
            f"⏱ Server time: <code>{server_ms}</code>\n"
            f"↔️ Selisih: <code>{diff} ms</code>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal ping: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /saldo
# ------------------------------------------------------------------
@restricted
async def cmd_saldo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        balances = bc.get_account_balance()
        aktif    = [b for b in balances if float(b.get("balance", 0)) != 0]
        if not aktif:
            await update.message.reply_text("💰 Saldo semua aset: 0")
            return
        lines = ["💰 <b>Saldo Akun Futures Demo</b>\n"]
        for b in aktif:
            lines.append(
                f"<b>{b['asset']}</b>\n"
                f"  Total    : <code>{_fmt(b['balance'])}</code>\n"
                f"  Tersedia : <code>{_fmt(b['availableBalance'])}</code>\n"
                f"  Unreal   : <code>{_fmt(b.get('crossUnPnl', 0))}</code>\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /akun
# ------------------------------------------------------------------
@restricted
async def cmd_akun(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        info = bc.get_account_info()
        text = (
            "📋 <b>Info Akun Futures Demo</b>\n\n"
            f"💵 Total Wallet   : <code>{_fmt(info.get('totalWalletBalance', 0), 2)} USDT</code>\n"
            f"📊 Unrealized PnL : <code>{_fmt(info.get('totalUnrealizedProfit', 0), 4)} USDT</code>\n"
            f"🏦 Margin Balance : <code>{_fmt(info.get('totalMarginBalance', 0), 2)} USDT</code>\n"
            f"✅ Tersedia       : <code>{_fmt(info.get('availableBalance', 0), 2)} USDT</code>\n"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /posisi [SYMBOL]
# ------------------------------------------------------------------
@restricted
async def cmd_posisi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    try:
        positions = bc.get_open_positions() if not symbol else [
            p for p in bc.get_position_risk(symbol)
            if float(p.get("positionAmt", 0)) != 0
        ]
        if not positions:
            await update.message.reply_text(
                f"📭 Tidak ada posisi aktif{' untuk ' + symbol if symbol else ''}.")
            return
        lines = [f"📈 <b>Posisi Aktif{' - ' + symbol if symbol else ''}</b>\n"]
        for p in positions:
            side = "🟢 LONG" if float(p["positionAmt"]) > 0 else "🔴 SHORT"
            lines.append(
                f"<b>{p['symbol']}</b> {side}\n"
                f"  Qty   : <code>{_fmt(p['positionAmt'])}</code>\n"
                f"  Entry : <code>{_fmt(p['entryPrice'])}</code>\n"
                f"  Mark  : <code>{_fmt(p['markPrice'])}</code>\n"
                f"  PnL   : <code>{_fmt(p['unRealizedProfit'])} USDT</code>\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /order [SYMBOL]
# ------------------------------------------------------------------
@restricted
async def cmd_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    try:
        orders = bc.get_open_orders(symbol)
        if not orders:
            await update.message.reply_text(
                f"📭 Tidak ada open order{' untuk ' + symbol if symbol else ''}.")
            return
        lines = [f"📋 <b>Open Orders{' - ' + symbol if symbol else ''}</b>\n"]
        for o in orders[:15]:
            side = "🟢 BUY" if o["side"] == "BUY" else "🔴 SELL"
            lines.append(
                f"<b>{o['symbol']}</b> {side}\n"
                f"  ID    : <code>{o['orderId']}</code>\n"
                f"  Qty   : <code>{_fmt(o['origQty'])}</code>\n"
                f"  Price : <code>{_fmt(o.get('price', 0))}</code>\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /riwayat SYMBOL
# ------------------------------------------------------------------
@restricted
async def cmd_riwayat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Gunakan: /riwayat BTCUSDT")
        return
    symbol = context.args[0].upper()
    limit  = int(context.args[1]) if len(context.args) > 1 else 10
    try:
        orders = bc.get_all_orders(symbol, limit=limit)
        if not orders:
            await update.message.reply_text(f"📭 Tidak ada riwayat untuk {symbol}.")
            return
        lines = [f"📜 <b>Riwayat Order - {symbol}</b>\n"]
        for o in reversed(orders):
            side = "🟢 BUY" if o["side"] == "BUY" else "🔴 SELL"
            lines.append(
                f"{side} <code>{o['type']}</code>\n"
                f"  ID     : <code>{o['orderId']}</code>\n"
                f"  Qty    : <code>{_fmt(o['origQty'])}</code>\n"
                f"  Status : <code>{o['status']}</code>\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /harga SYMBOL
# ------------------------------------------------------------------
@restricted
async def cmd_harga(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Gunakan: /harga BTCUSDT")
        return
    symbol = context.args[0].upper()
    try:
        data = bc.get_ticker_price(symbol)
        await update.message.reply_text(
            f"💲 <b>{symbol}</b> : <code>{_fmt(data['price'])} USDT</code>",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /24jam SYMBOL
# ------------------------------------------------------------------
@restricted
async def cmd_24jam(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Gunakan: /24jam BTCUSDT")
        return
    symbol = context.args[0].upper()
    try:
        d     = bc.get_24hr_ticker(symbol)
        emoji = "📈" if float(d["priceChangePercent"]) >= 0 else "📉"
        text  = (
            f"{emoji} <b>{symbol} — 24 Jam</b>\n\n"
            f"Last   : <code>{_fmt(d['lastPrice'], 2)}</code>\n"
            f"High   : <code>{_fmt(d['highPrice'], 2)}</code>\n"
            f"Low    : <code>{_fmt(d['lowPrice'], 2)}</code>\n"
            f"Change : <code>{_fmt(d['priceChangePercent'], 2)}%</code>\n"
            f"Volume : <code>{_fmt(d['volume'], 2)}</code>\n"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# /scan — Lihat skor top 100 token tanpa threshold
# ------------------------------------------------------------------
@restricted
async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Scanning top 100 token, harap tunggu 3–5 menit...",
        parse_mode=ParseMode.HTML,
    )

    import functools
    from scanner import scan
    from config  import DEFAULT_INTERVAL

    loop = asyncio.get_event_loop()

    try:
        results = await loop.run_in_executor(
            None,
            functools.partial(scan, top_n=100, interval=DEFAULT_INTERVAL, threshold=0.0)
        )

        if not results:
            await update.message.reply_text("❌ Tidak ada hasil scan.")
            return

        # Kirim per batch 25 token
        batch_size = 25
        for batch_num, i in enumerate(range(0, min(len(results), 100), batch_size), 1):
            batch = results[i:i + batch_size]
            lines = [f"📊 <b>Scan Result — Batch {batch_num}/4</b>\n"]
            for rank, r in enumerate(batch, i + 1):
                sym   = r["symbol"]
                total = r["weighted_total"]
                dir_  = r["direction"]
                emoji = "🟢" if dir_ == "LONG" else ("🔴" if dir_ == "SHORT" else "⚪")
                # Tampilkan juga breakdown skor penting
                s = r["scores"]
                lines.append(
                    f"{rank:>3}. {emoji} <b>{sym:<14}</b> "
                    f"<code>{total:+.2f}</code>  "
                    f"f={s.get('funding', 0):+.0f} "
                    f"lsr={s.get('lsr', 0):+.0f} "
                    f"rsi={s.get('rsi', 0):+.0f} "
                    f"ma={s.get('ma', 0):+.0f}"
                )
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
            await asyncio.sleep(0.3)

    except Exception as e:
        await update.message.reply_text(f"❌ Error: <code>{e}</code>", parse_mode=ParseMode.HTML)


# ------------------------------------------------------------------
# Build Application
# ------------------------------------------------------------------
def build_application() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("ping",    cmd_ping))
    app.add_handler(CommandHandler("saldo",   cmd_saldo))
    app.add_handler(CommandHandler("akun",    cmd_akun))
    app.add_handler(CommandHandler("posisi",  cmd_posisi))
    app.add_handler(CommandHandler("order",   cmd_order))
    app.add_handler(CommandHandler("riwayat", cmd_riwayat))
    app.add_handler(CommandHandler("harga",   cmd_harga))
    app.add_handler(CommandHandler("24jam",   cmd_24jam))
    app.add_handler(CommandHandler("scan",    cmd_scan))

    async def post_init(application: Application) -> None:
        from scheduler import setup_scheduler
        scheduler = setup_scheduler(application.bot)
        scheduler.start()
        logger.info("[telegram_bot] Scheduler started")

    app.post_init = post_init

    return app
