import logging
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

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


# ------------------------------------------------------------------
# Helper: format angka
# ------------------------------------------------------------------
def _fmt(val, decimals=4) -> str:
    try:
        return f"{float(val):,.{decimals}f}"
    except Exception:
        return str(val)


# ------------------------------------------------------------------
# /start - Menu utama
# ------------------------------------------------------------------
@restricted
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Binance Futures Demo Bot*\n\n"
        "Perintah yang tersedia:\n"
        "📊 *Akun & Saldo*\n"
        "  /saldo — Cek saldo akun\n"
        "  /akun — Info akun lengkap\n\n"
        "📈 *Posisi*\n"
        "  /posisi — Semua posisi aktif\n"
        "  /posisi BTCUSDT — Posisi simbol tertentu\n\n"
        "📋 *Order*\n"
        "  /order — Semua open order\n"
        "  /order BTCUSDT — Open order simbol tertentu\n"
        "  /riwayat BTCUSDT — 10 order terakhir\n\n"
        "💰 *Harga*\n"
        "  /harga BTCUSDT — Harga terakhir\n"
        "  /24jam BTCUSDT — Statistik 24 jam\n\n"
        "🔧 *Lainnya*\n"
        "  /ping — Cek koneksi ke Binance\n"
        "  /help — Tampilkan menu ini\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


cmd_help = cmd_start  # /help sama dengan /start


# ------------------------------------------------------------------
# /ping - Cek koneksi
# ------------------------------------------------------------------
@restricted
async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        server_ms = bc.get_server_time()
        import time
        local_ms = int(time.time() * 1000)
        diff = local_ms - server_ms
        await update.message.reply_text(
            f"✅ Terhubung ke Binance Testnet\n"
            f"⏱ Server time: `{server_ms}`\n"
            f"↔️ Selisih lokal-server: `{diff} ms`",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal ping: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ------------------------------------------------------------------
# /saldo - Cek saldo
# ------------------------------------------------------------------
@restricted
async def cmd_saldo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        balances = bc.get_account_balance()
        # Filter aset yang ada saldo
        aktif = [b for b in balances if float(b.get("balance", 0)) != 0]
        if not aktif:
            await update.message.reply_text("💰 Saldo semua aset: 0")
            return

        lines = ["💰 *Saldo Akun Futures Demo*\n"]
        for b in aktif:
            asset = b["asset"]
            balance = _fmt(b["balance"], 4)
            available = _fmt(b["availableBalance"], 4)
            unrealized = _fmt(b.get("crossUnPnl", 0), 4)
            lines.append(
                f"*{asset}*\n"
                f"  Total      : `{balance}`\n"
                f"  Tersedia   : `{available}`\n"
                f"  Unrealized : `{unrealized}`\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ------------------------------------------------------------------
# /akun - Info akun lengkap
# ------------------------------------------------------------------
@restricted
async def cmd_akun(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        info = bc.get_account_info()
        total_wallet = _fmt(info.get("totalWalletBalance", 0), 2)
        total_unrealized = _fmt(info.get("totalUnrealizedProfit", 0), 4)
        total_margin = _fmt(info.get("totalMarginBalance", 0), 2)
        total_maint = _fmt(info.get("totalMaintMargin", 0), 4)
        available = _fmt(info.get("availableBalance", 0), 2)
        can_deposit = info.get("canDeposit", False)
        can_trade = info.get("canTrade", False)
        can_withdraw = info.get("canWithdraw", False)

        text = (
            "📋 *Info Akun Futures Demo*\n\n"
            f"💵 Total Wallet   : `{total_wallet} USDT`\n"
            f"📊 Unrealized PnL : `{total_unrealized} USDT`\n"
            f"🏦 Margin Balance : `{total_margin} USDT`\n"
            f"🔒 Maint Margin   : `{total_maint} USDT`\n"
            f"✅ Tersedia       : `{available} USDT`\n\n"
            f"⚙️ Deposit  : {'✅' if can_deposit else '❌'} | "
            f"Trade   : {'✅' if can_trade else '❌'} | "
            f"Withdraw: {'✅' if can_withdraw else '❌'}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ------------------------------------------------------------------
# /posisi [SYMBOL] - Cek posisi aktif
# ------------------------------------------------------------------
@restricted
async def cmd_posisi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    try:
        if symbol:
            positions = bc.get_position_risk(symbol)
            positions = [p for p in positions if float(p.get("positionAmt", 0)) != 0]
        else:
            positions = bc.get_open_positions()

        if not positions:
            msg = f"📭 Tidak ada posisi aktif{' untuk ' + symbol if symbol else ''}."
            await update.message.reply_text(msg)
            return

        lines = [f"📈 *Posisi Aktif{' - ' + symbol if symbol else ''}*\n"]
        for p in positions:
            sym = p["symbol"]
            amt = _fmt(p["positionAmt"], 4)
            entry = _fmt(p["entryPrice"], 4)
            mark = _fmt(p["markPrice"], 4)
            pnl = _fmt(p["unRealizedProfit"], 4)
            leverage = p.get("leverage", "?")
            side = "🟢 LONG" if float(p["positionAmt"]) > 0 else "🔴 SHORT"

            lines.append(
                f"*{sym}* {side}\n"
                f"  Qty        : `{amt}`\n"
                f"  Entry      : `{entry}`\n"
                f"  Mark Price : `{mark}`\n"
                f"  Leverage   : `{leverage}x`\n"
                f"  PnL        : `{pnl} USDT`\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ------------------------------------------------------------------
# /order [SYMBOL] - Cek open order
# ------------------------------------------------------------------
@restricted
async def cmd_order(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = context.args[0].upper() if context.args else None
    try:
        orders = bc.get_open_orders(symbol)
        if not orders:
            msg = f"📭 Tidak ada open order{' untuk ' + symbol if symbol else ''}."
            await update.message.reply_text(msg)
            return

        lines = [f"📋 *Open Orders{' - ' + symbol if symbol else ''}*\n"]
        for o in orders[:15]:  # batasi 15 agar tidak terlalu panjang
            sym = o["symbol"]
            side = "🟢 BUY" if o["side"] == "BUY" else "🔴 SELL"
            otype = o["type"]
            qty = _fmt(o["origQty"], 4)
            price = _fmt(o.get("price", 0), 4)
            status = o["status"]
            oid = o["orderId"]

            lines.append(
                f"*{sym}* {side} `{otype}`\n"
                f"  OrderID : `{oid}`\n"
                f"  Qty     : `{qty}`\n"
                f"  Price   : `{price}`\n"
                f"  Status  : `{status}`\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ------------------------------------------------------------------
# /riwayat SYMBOL - 10 order terakhir
# ------------------------------------------------------------------
@restricted
async def cmd_riwayat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Gunakan: /riwayat BTCUSDT")
        return
    symbol = context.args[0].upper()
    limit = int(context.args[1]) if len(context.args) > 1 else 10
    try:
        orders = bc.get_all_orders(symbol, limit=limit)
        if not orders:
            await update.message.reply_text(f"📭 Tidak ada riwayat order untuk {symbol}.")
            return

        lines = [f"📜 *Riwayat Order - {symbol}* (terakhir {limit})\n"]
        for o in reversed(orders):  # terbaru duluan
            side = "🟢 BUY" if o["side"] == "BUY" else "🔴 SELL"
            otype = o["type"]
            qty = _fmt(o["origQty"], 4)
            price = _fmt(o.get("avgPrice") or o.get("price", 0), 4)
            status = o["status"]
            oid = o["orderId"]

            lines.append(
                f"{side} `{otype}` — *{sym if (sym := symbol) else symbol}*\n"
                f"  ID     : `{oid}`\n"
                f"  Qty    : `{qty}`\n"
                f"  Price  : `{price}`\n"
                f"  Status : `{status}`\n"
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ------------------------------------------------------------------
# /harga SYMBOL - Harga terakhir
# ------------------------------------------------------------------
@restricted
async def cmd_harga(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Gunakan: /harga BTCUSDT")
        return
    symbol = context.args[0].upper()
    try:
        data = bc.get_ticker_price(symbol)
        price = _fmt(data["price"], 4)
        await update.message.reply_text(
            f"💲 *{symbol}* : `{price} USDT`",
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)


# ------------------------------------------------------------------
# /24jam SYMBOL - Statistik 24 jam
# ------------------------------------------------------------------
@restricted
async def cmd_24jam(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Gunakan: /24jam BTCUSDT")
        return
    symbol = context.args[0].upper()
    try:
        d = bc.get_24hr_ticker(symbol)
        last = _fmt(d["lastPrice"], 2)
        high = _fmt(d["highPrice"], 2)
        low = _fmt(d["lowPrice"], 2)
        change = _fmt(d["priceChangePercent"], 2)
        volume = _fmt(d["volume"], 2)
        q_volume = _fmt(d["quoteVolume"], 2)

        emoji = "📈" if float(d["priceChangePercent"]) >= 0 else "📉"
        text = (
            f"{emoji} *{symbol} — Statistik 24 Jam*\n\n"
            f"Last Price : `{last}`\n"
            f"High       : `{high}`\n"
            f"Low        : `{low}`\n"
            f"Change     : `{change}%`\n"
            f"Volume     : `{volume}`\n"
            f"Quote Vol  : `{q_volume} USDT`"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode=ParseMode.MARKDOWN)
 
        
@restricted
async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "⚠️ Gunakan: /analyze BTCUSDT\n"
            "           /analyze BTCUSDT 1h",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
 
    symbol   = context.args[0].upper()
    interval = context.args[1] if len(context.args) > 1 else "30m"
 
    # notify callback: kirim pesan ke chat yang sama
    async def _send(msg: str):
        await update.message.reply_text(msg, parse_mode="HTML")
 
    # Karena pipeline sync, jalankan di thread terpisah
    import asyncio
    from pipeline import run as run_pipeline
 
    await update.message.reply_text(
        f"🚀 <b>Memulai analisis {symbol} {interval}...</b>",
        parse_mode="HTML",
    )
 
    loop = asyncio.get_event_loop()
    # Buat notify yang thread-safe
    import functools
 
    def sync_notify(msg: str):
        # schedule coroutine ke event loop
        asyncio.run_coroutine_threadsafe(
            update.message.reply_text(msg, parse_mode="HTML"),
            loop,
        )
 
    try:
        result = await loop.run_in_executor(
            None,
            functools.partial(run_pipeline, symbol, interval, sync_notify),
        )
        if result.get("stage") == "completed":
            pass  # semua notif sudah dikirim via sync_notify
        elif result.get("skipped"):
            pass  # skip notif sudah dikirim
    except Exception as e:
        await update.message.reply_text(
            f"❌ Pipeline error: <code>{e}</code>",
            parse_mode="HTML",
        )


# ------------------------------------------------------------------
# Fungsi builder: return Application yang siap jalan
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
    app.add_handler(CommandHandler("analyze", cmd_analyze))

    return app
