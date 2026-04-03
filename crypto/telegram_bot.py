# =============================================================
# telegram_bot.py — Telegram bot handler
# =============================================================

import asyncio
import logging

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
    ConversationHandler, MessageHandler, filters,
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
        "📝 <b>Paper Trading</b>\n"
        "  /paper_status — Posisi aktif + track record simulasi\n"
        "  /24jam BTCUSDT — Statistik 24 jam\n\n"
        "🔍 <b>Scan</b>\n"
        "  /scan — Lihat skor semua top 50 token\n\n"
        "🔧 <b>Lainnya</b>\n"
        "  /ping — Cek koneksi ke Binance\n"
        "  /buatorder — Buat order manual (interaktif)\n"
        "  /batal — Batalkan order yang sedang dibuat\n"
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
# /scan — Lihat skor top 50 token + BTC context
# ------------------------------------------------------------------
@restricted
async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Scanning BTC + top 50 token, harap tunggu 3–5 menit...",
        parse_mode=ParseMode.HTML,
    )

    import functools
    from scanner import scan
    from config  import DEFAULT_INTERVAL

    loop = asyncio.get_event_loop()

    try:
        results = await loop.run_in_executor(
            None,
            functools.partial(scan, top_n=50, interval=DEFAULT_INTERVAL, threshold=0.0)
        )

        if not results:
            await update.message.reply_text("❌ Tidak ada hasil scan.")
            return

        # Ambil BTC multiplier info dari result pertama
        btc_mult  = results[0].get("btc_multiplier_applied", 0.0) if results else 0.0
        btc_label = (
            f"🟢 Bullish (×{abs(btc_mult):.1f})" if btc_mult > 0 else
            f"🔴 Bearish (×{abs(btc_mult):.1f})" if btc_mult < 0 else
            "⚪ Netral"
        )

        # Header BTC context
        await update.message.reply_text(
            f"₿ <b>BTC Market Context</b> : {btc_label}",
            parse_mode=ParseMode.HTML,
        )

        # Kirim per batch 25 token
        batch_size = 25
        total_show = min(len(results), 50)
        n_batches  = (total_show + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, total_show, batch_size), 1):
            batch = results[i:i + batch_size]
            lines = [f"📊 <b>Scan Result — Batch {batch_num}/{n_batches}</b>\n"]
            for rank, r in enumerate(batch, i + 1):
                sym   = r["symbol"]
                total = r["weighted_total"]
                raw   = r.get("weighted_total_raw", total)
                dir_  = r["direction"]
                emoji = "🟢" if dir_ == "LONG" else ("🔴" if dir_ == "SHORT" else "⚪")
                s     = r["scores"]
                lines.append(
                    f"{rank:>3}. {emoji} <b>{sym:<14}</b> "
                    f"<code>{total:+.2f}</code> "
                    f"<i>(raw {raw:+.2f})</i>  "
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
# /buatorder — Manual order interaktif
# States: SYMBOL → SIDE → ENTRY → SL → TP → LEVERAGE → MODAL → CONFIRM
# ------------------------------------------------------------------
_BO_SYMBOL, _BO_SIDE, _BO_ENTRY, _BO_SL, _BO_TP, _BO_LEVERAGE_S, _BO_MODAL, _BO_CONFIRM = range(8)


@restricted
async def cmd_buatorder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    from config import PAPER_TRADING_MODE
    mode_tag = "PAPER" if PAPER_TRADING_MODE else "REAL"
    await update.message.reply_text(
        f"📝 <b>Buat Order Manual [{mode_tag}]</b>\n\nMasukkan <b>symbol</b> (contoh: ENAUSDT):",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove(),
    )
    return _BO_SYMBOL


async def _bo_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = update.message.text.strip().upper()
    if not symbol.endswith("USDT"):
        await update.message.reply_text("⚠️ Symbol harus diakhiri USDT. Coba lagi:")
        return _BO_SYMBOL
    context.user_data["symbol"] = symbol
    await update.message.reply_text(
        f"✅ Symbol: <b>{symbol}</b>\n\nSide?",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardMarkup([["BUY", "SELL"]], one_time_keyboard=True, resize_keyboard=True),
    )
    return _BO_SIDE


async def _bo_side(update: Update, context: ContextTypes.DEFAULT_TYPE):
    side = update.message.text.strip().upper()
    if side not in ("BUY", "SELL"):
        await update.message.reply_text("⚠️ Pilih BUY atau SELL:")
        return _BO_SIDE
    context.user_data["side"] = side
    await update.message.reply_text(
        f"✅ Side: <b>{side}</b>\n\nMasukkan <b>Entry Price</b>:",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove(),
    )
    return _BO_ENTRY


async def _bo_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        entry = float(update.message.text.strip())
        assert entry > 0
    except Exception:
        await update.message.reply_text("⚠️ Harga tidak valid. Masukkan angka positif:")
        return _BO_ENTRY
    context.user_data["entry"] = entry
    side = context.user_data["side"]
    hint = "di bawah entry" if side == "BUY" else "di atas entry"
    await update.message.reply_text(
        f"✅ Entry: <b>{entry}</b>\n\nMasukkan <b>Stop Loss</b> ({hint}):",
        parse_mode=ParseMode.HTML,
    )
    return _BO_SL


async def _bo_sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sl = float(update.message.text.strip())
        assert sl > 0
    except Exception:
        await update.message.reply_text("⚠️ Harga tidak valid. Masukkan angka positif:")
        return _BO_SL

    entry = context.user_data["entry"]
    side  = context.user_data["side"]

    if side == "BUY" and sl >= entry:
        await update.message.reply_text(f"⚠️ SL harus di bawah entry ({entry}). Coba lagi:")
        return _BO_SL
    if side == "SELL" and sl <= entry:
        await update.message.reply_text(f"⚠️ SL harus di atas entry ({entry}). Coba lagi:")
        return _BO_SL

    context.user_data["sl"] = sl
    hint = "di atas entry" if side == "BUY" else "di bawah entry"
    await update.message.reply_text(
        f"✅ SL: <b>{sl}</b>\n\nMasukkan <b>Take Profit</b> ({hint}):",
        parse_mode=ParseMode.HTML,
    )
    return _BO_TP


async def _bo_tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        tp = float(update.message.text.strip())
        assert tp > 0
    except Exception:
        await update.message.reply_text("⚠️ Harga tidak valid. Masukkan angka positif:")
        return _BO_TP

    entry = context.user_data["entry"]
    side  = context.user_data["side"]

    if side == "BUY" and tp <= entry:
        await update.message.reply_text(f"⚠️ TP harus di atas entry ({entry}). Coba lagi:")
        return _BO_TP
    if side == "SELL" and tp >= entry:
        await update.message.reply_text(f"⚠️ TP harus di bawah entry ({entry}). Coba lagi:")
        return _BO_TP

    context.user_data["tp"] = tp
    await update.message.reply_text(
        f"✅ TP: <b>{tp}</b>\n\nMasukkan <b>Leverage</b> (angka, contoh: 10):",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardMarkup([["5", "10", "15", "20"]], one_time_keyboard=True, resize_keyboard=True),
    )
    return _BO_LEVERAGE_S


async def _bo_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        lev = int(update.message.text.strip())
        assert 1 <= lev <= 50
    except Exception:
        await update.message.reply_text("⚠️ Leverage tidak valid. Masukkan angka 1-50:")
        return _BO_LEVERAGE_S
    context.user_data["leverage"] = lev
    await update.message.reply_text(
        f"✅ Leverage: <b>{lev}x</b>\n\nMasukkan <b>Modal</b> dalam USDT (contoh: 100):",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove(),
    )
    return _BO_MODAL


async def _bo_modal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        modal = float(update.message.text.strip())
        assert modal > 0
    except Exception:
        await update.message.reply_text("⚠️ Modal tidak valid. Masukkan angka USDT positif:")
        return _BO_MODAL

    context.user_data["modal"] = modal
    d     = context.user_data
    entry = d["entry"]
    sl    = d["sl"]
    tp    = d["tp"]
    lev   = d["leverage"]
    rr    = abs(tp - entry) / abs(entry - sl)

    # Hitung notional & PnL estimasi
    notional = modal * lev
    if d["side"] == "BUY":
        pnl_tp = (tp - entry) / entry * notional
        pnl_sl = (sl - entry) / entry * notional
    else:
        pnl_tp = (entry - tp) / entry * notional
        pnl_sl = (entry - sl) / entry * notional

    from config import PAPER_TRADING_MODE
    mode_tag = "PAPER" if PAPER_TRADING_MODE else "REAL"

    await update.message.reply_text(
        f"📋 <b>Konfirmasi Order [{mode_tag}]</b>\n"
        f"─────────────────────\n"
        f"  Symbol   : <b>{d['symbol']}</b>\n"
        f"  Side     : <b>{d['side']}</b>\n"
        f"  Entry    : <code>{entry}</code>\n"
        f"  SL       : <code>{sl}</code>\n"
        f"  TP       : <code>{tp}</code>\n"
        f"  RR       : <code>{rr:.2f}</code>\n"
        f"  Leverage : <b>{lev}x</b>\n"
        f"  Modal    : <code>{modal} USDT</code>\n"
        f"  Notional : <code>{notional:.2f} USDT</code>\n"
        f"  Est TP   : <b>+{pnl_tp:.2f} USDT</b>\n"
        f"  Est SL   : <b>{pnl_sl:.2f} USDT</b>\n\n"
        f"Lanjut kirim order?",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardMarkup([["✅ YA, KIRIM", "❌ BATAL"]], one_time_keyboard=True, resize_keyboard=True),
    )
    return _BO_CONFIRM


async def _bo_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if "BATAL" in text:
        await update.message.reply_text("❌ Order dibatalkan.", reply_markup=ReplyKeyboardRemove())
        context.user_data.clear()
        return ConversationHandler.END

    d      = context.user_data
    symbol = d["symbol"]
    side   = d["side"]
    action = "BUYING" if side == "BUY" else "SELLING"
    modal  = d["modal"]
    lev    = d["leverage"]

    from config import PAPER_TRADING_MODE

    # qty_fraction = modal / PAPER_BALANCE (paper) atau modal / real_balance (real)
    # Kita pakai modal langsung sebagai notional override via qty_fraction trick:
    # qty_fraction = modal / balance, tapi di paper kita override langsung
    # Simpan modal di ai_result sebagai custom field
    ai_result = {
        "action":       action,
        "entry_price":  d["entry"],
        "stop_loss":    d["sl"],
        "take_profit":  d["tp"],
        "leverage":     lev,
        "qty_fraction": 1.0,   # akan di-override di bawah
        "_modal_usdt":  modal,  # custom field untuk paper executor
    }
    pred = {"symbol": symbol}

    await update.message.reply_text(
        f"📤 Mengirim order {side} <b>{symbol}</b>...",
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardRemove(),
    )

    try:
        import asyncio as _asyncio
        loop    = _asyncio.get_event_loop()
        chat_id = update.effective_chat.id
        bot     = update.get_bot()

        def _notify(msg: str):
            _asyncio.run_coroutine_threadsafe(
                bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.HTML),
                loop,
            )

        if PAPER_TRADING_MODE:
            from order.paper_executor import execute_paper_order
            # Override qty_fraction berdasarkan modal yang diinput user
            # notional = modal * lev, qty_fraction = modal / PAPER_BALANCE_USDT
            from config import PAPER_BALANCE_USDT
            ai_result["qty_fraction"] = min(modal / PAPER_BALANCE_USDT, 1.0)
            result = await loop.run_in_executor(
                None,
                lambda: execute_paper_order(ai_result, pred, notify_fn=_notify),
            )
        else:
            from order.executor import execute_order
            # Real: qty_fraction = modal / available_balance
            # Tapi kita set leverage dan biarkan executor hitung qty dari notional
            # Trick: set qty_fraction = modal/balance — executor fetch balance sendiri
            # Simpel: pakai modal sebagai notional langsung, fraction = modal/MAX_NOTIONAL
            from order.executor import MAX_NOTIONAL_USDT
            ai_result["qty_fraction"] = min(modal / MAX_NOTIONAL_USDT, 1.0)
            result = await loop.run_in_executor(
                None,
                lambda: execute_order(ai_result, pred, notify_fn=_notify),
            )

        if result.get("ok"):
            paper_tag = " PAPER" if result.get("paper") else ""
            await update.message.reply_text(
                f"✅ <b>Order{paper_tag} Terkirim — {symbol}</b>\n"
                f"  ID       : <code>{result['order_id']}</code>\n"
                f"  Side     : <b>{result['side']}</b>\n"
                f"  Entry    : <code>{result['entry_price']}</code>\n"
                f"  SL       : <code>{result['stop_loss']}</code>\n"
                f"  TP       : <code>{result['take_profit']}</code>\n"
                f"  Leverage : <b>{result['leverage']}x</b>\n"
                f"  Margin   : <code>{result['balance_used']} USDT</code>",
                parse_mode=ParseMode.HTML,
            )
        else:
            reason = str(result.get("reason_fail", "unknown"))
            await update.message.reply_text(
                f"❌ Order gagal: <code>{reason}</code>",
                parse_mode=ParseMode.HTML,
            )

    except Exception as e:
        await update.message.reply_text(
            f"❌ Error: <code>{str(e)}</code>",
            parse_mode=ParseMode.HTML,
        )

    context.user_data.clear()
    return ConversationHandler.END


async def _bo_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Order dibatalkan.", reply_markup=ReplyKeyboardRemove())
    context.user_data.clear()
    return ConversationHandler.END


# ------------------------------------------------------------------
# /paper_status — Ringkasan paper trading
# ------------------------------------------------------------------
@restricted
async def cmd_paper_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        from order.paper_watcher import get_paper_status, format_paper_status_telegram
        status = get_paper_status()
        text   = format_paper_status_telegram(status)
    except ImportError as e:
        text = (
            f"⚠️ <b>Paper watcher belum tersedia</b>\n"
            f"<code>{e}</code>\n\n"
            f"Pastikan file <code>order/paper_watcher.py</code> sudah ada di server."
        )
    except Exception as e:
        text = f"❌ <b>Error:</b> <code>{e}</code>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


def _build_buatorder_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[CommandHandler("buatorder", cmd_buatorder)],
        states={
            _BO_SYMBOL:     [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_symbol)],
            _BO_SIDE:       [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_side)],
            _BO_ENTRY:      [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_entry)],
            _BO_SL:         [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_sl)],
            _BO_TP:         [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_tp)],
            _BO_LEVERAGE_S: [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_leverage)],
            _BO_MODAL:      [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_modal)],
            _BO_CONFIRM:    [MessageHandler(filters.TEXT & ~filters.COMMAND, _bo_confirm)],
        },
        fallbacks=[CommandHandler("batal", _bo_cancel)],
        allow_reentry=True,
    )


# ------------------------------------------------------------------
# Build Application
# ------------------------------------------------------------------
def build_application() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(_build_buatorder_handler())
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
    app.add_handler(CommandHandler("scan",         cmd_scan))
    app.add_handler(CommandHandler("paper_status", cmd_paper_status))

    async def post_init(application: Application) -> None:
        from scheduler import setup_scheduler
        from config import PAPER_TRADING_MODE
        scheduler = setup_scheduler(application.bot)
        scheduler.start()
        logger.info("[telegram_bot] Scheduler started")

        if PAPER_TRADING_MODE:
            try:
                from order.paper_watcher import start_paper_watcher
                def _tg_notify(msg: str):
                    import asyncio
                    for chat_id in ALLOWED_CHAT_IDS:
                        asyncio.create_task(
                            application.bot.send_message(
                                chat_id=chat_id, text=msg, parse_mode=ParseMode.HTML
                            )
                        )
                start_paper_watcher(notify_fn=_tg_notify)
                logger.info("[telegram_bot] Paper watcher started")
            except Exception as e:
                logger.error("[telegram_bot] Gagal start paper watcher: %s", e)

    app.post_init = post_init

    return app
