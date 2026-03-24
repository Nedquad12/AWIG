"""
formatter.py — Format pesan Telegram untuk crypto bot
"""

from datetime import datetime


def fmt_pipeline_result(r: dict) -> str:
    """Format hasil full pipeline jadi pesan Telegram HTML."""
    symbol = r.get("symbol", "?")
    error  = r.get("error")

    if error and r.get("stage") not in ("trade", "done"):
        return f"❌ <b>{symbol}</b> — Error di tahap [{r.get('stage','?')}]\n{error}"

    lines = [f"📊 <b>{symbol} — Hasil Analisis</b>", ""]

    # Fetch info
    fetch = r.get("fetch")
    if fetch:
        lines.append(f"📥 Data   : {fetch['candles']} candle | terakhir {fetch['last']}")

    # Scores
    sc = r.get("scores")
    if sc:
        arrow = "🟢" if sc["total"] > 0 else "🔴"
        lines += [
            "",
            "<b>── Skor Indikator ──</b>",
            f"{arrow} Total : <b>{sc['total']:+.2f}</b>  |  Harga: <b>{sc['price']:.6f}</b> ({sc['change']:+.4f}%)",
            f"<code>VSA:{sc['vsa']:+d}  FSA:{sc['fsa']:+d}  VFA:{sc['vfa']:+d}  WCC:{sc['wcc']:+d}  SRST:{sc['srst']:+d}</code>",
            f"<code>RSI:{sc['rsi']:+d}  MACD:{sc['macd']:+d}  MA:{sc['ma']:+d}  IP:{sc['ip_score']:+.1f}  Tight:{sc['tight']:+d}</code>",
        ]

    # ML Adj
    ml_adj = r.get("ml_adj")
    if ml_adj and ml_adj.get("success"):
        delta = ml_adj["accuracy_after"] - ml_adj["accuracy_before"]
        arr   = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        lines += [
            "",
            "<b>── ML Weight Adj ──</b>",
            f"Akurasi: {ml_adj['accuracy_before']:.1f}% → <b>{ml_adj['accuracy_after']:.1f}%</b> {arr}{abs(delta):.1f}%  ({ml_adj['n_bars']} bar)",
        ]
    elif ml_adj and not ml_adj.get("success"):
        lines.append(f"⚠️ ML Adj: {ml_adj.get('message','skip')}")

    # ML Pred
    ml_pred = r.get("ml_pred")
    if ml_pred and not ml_pred.get("error"):
        emoji = {"NAIK": "🟢", "TURUN": "🔴", "NETRAL": "⚪"}.get(ml_pred.get("label",""), "❓")
        sl_s  = ml_pred.get("sl_pct", 0)
        tp_s  = ml_pred.get("tp_pct", 0)
        rr    = f"{tp_s/sl_s:.1f}x" if sl_s > 0 else "N/A"
        lines += [
            "",
            "<b>── Prediksi ML (3 candle ke depan) ──</b>",
            f"{emoji} <b>{ml_pred.get('label','?')}</b>  conf: <b>{ml_pred.get('confidence',0):.1f}%</b>  win_rate: {ml_pred.get('win_rate',0):.1f}%",
            f"Proba → Naik: {ml_pred.get('proba_up',0):.1f}%  Netral: {ml_pred.get('proba_flat',0):.1f}%  Turun: {ml_pred.get('proba_down',0):.1f}%",
            f"SL saran: <code>{sl_s:.2f}%</code>  TP saran: <code>{tp_s:.2f}%</code>  R/R: <b>{rr}</b>",
        ]
    elif ml_pred and ml_pred.get("error"):
        lines.append(f"⚠️ ML Pred: {ml_pred['error']}")

    # AI Decision
    ai = r.get("ai")
    if ai and not ai.get("error"):
        dec       = ai.get("decision","?")
        dec_emoji = "✅" if dec == "BUY" else "⏭"
        lines += ["", "<b>── Keputusan AI ──</b>", f"{dec_emoji} <b>{dec}</b>"]
        if dec == "BUY":
            sl_pct = ai.get("sl_pct", 0)
            tp_pct = ai.get("tp_pct", 0)
            rr     = f"{tp_pct/sl_pct:.1f}x" if sl_pct > 0 else "N/A"
            lines += [
                f"   Arah      : <b>{ai.get('direction','?')}</b>",
                f"   Leverage  : <b>{ai.get('leverage',0)}x</b>",
                f"   Modal     : <b>{ai.get('capital_pct',0)*100:.0f}%</b>",
                f"   Confidence: <b>{ai.get('confidence',0)}%</b>",
                f"   Stop Loss : <code>{sl_pct:.2f}%</code>  dari entry",
                f"   Take Profit: <code>{tp_pct:.2f}%</code>  dari entry  (R/R <b>{rr}</b>)",
            ]
        lines.append(f"   Alasan: <i>{ai.get('reason','')}</i>")
    elif ai and ai.get("error"):
        lines.append(f"❌ AI Error: {ai['error']}")

    # Trade result
    trade = r.get("trade")
    if trade:
        if trade.get("success"):
            lines += [
                "",
                "<b>── Eksekusi Order ──</b>",
                f"✅ {trade.get('direction','')} {trade.get('quantity','')} {symbol} @ <b>{trade.get('entry_price',0):.4f}</b>",
                f"   Leverage : <b>{trade.get('leverage',0)}x</b>",
                f"   SL       : <code>{trade.get('sl_price',0):.4f}</code>  (-{trade.get('sl_pct',0):.2f}%)",
                f"   TP       : <code>{trade.get('tp_price',0):.4f}</code>  (+{trade.get('tp_pct',0):.2f}%)",
                f"   Entry ID : <code>{trade.get('entry_order_id','?')}</code>",
                f"   SL ID    : <code>{trade.get('sl_order_id','?')}</code>",
                f"   TP ID    : <code>{trade.get('tp_order_id','?')}</code>",
            ]
        else:
            lines.append(f"\n⏭ {trade.get('message','Skip')}")

    if error:
        lines.append(f"\n⚠️ Error: {error}")

    lines.append(f"\n<i>{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</i>")
    return "\n".join(lines)


def fmt_scores_only(sc: dict) -> str:
    """Format skor saja (untuk /skor command)."""
    if sc.get("error"):
        return f"❌ {sc['error']}"
    arrow = "🟢" if sc["total"] > 0 else "🔴"
    return (
        f"📊 <b>{sc['symbol']}</b> — Quick Score\n\n"
        f"{arrow} Total : <b>{sc['total']:+.2f}</b>\n"
        f"Harga  : <b>{sc['price']:.6f}</b> ({sc['change']:+.4f}%)\n\n"
        f"<pre>"
        f"VSA    : {sc['vsa']:+d}\n"
        f"FSA    : {sc['fsa']:+d}\n"
        f"VFA    : {sc['vfa']:+d}\n"
        f"WCC    : {sc['wcc']:+d}\n"
        f"SRST   : {sc['srst']:+d}\n"
        f"RSI    : {sc['rsi']:+d}\n"
        f"MACD   : {sc['macd']:+d}\n"
        f"MA     : {sc['ma']:+d}\n"
        f"IP Raw : {sc['ip_raw']:.2f}\n"
        f"IP Skor: {sc['ip_score']:+.1f}\n"
        f"Tight  : {sc['tight']:+d}\n"
        f"──────────\n"
        f"TOTAL  : {sc['total']:+.2f}"
        f"</pre>"
    )


def fmt_positions(positions: list[dict], local_state: dict = None) -> str:
    """
    Format daftar posisi terbuka dari Binance + SL/TP info dari local state.

    Args:
        positions   : list dari get_open_positions() (Binance)
        local_state : dict dari load_positions() (untuk SL/TP data)
    """
    if not positions:
        return "📭 Tidak ada posisi terbuka."

    local_state = local_state or {}
    lines = ["📋 <b>Posisi Terbuka:</b>", ""]

    for p in positions:
        sym   = p["symbol"]
        pnl   = p.get("unrealizedProfit", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        local = local_state.get(sym, {})

        lines.append(f"{emoji} <b>{sym}</b>  {p['side']}")
        lines.append(f"   Qty    : {p['positionAmt']}")
        lines.append(f"   Entry  : {p['entryPrice']}")
        lines.append(f"   Mark   : {p.get('markPrice', '?')}")
        lines.append(f"   PnL    : <b>{pnl:+.4f} USDT</b>")
        lines.append(f"   Lev    : {p['leverage']}x")

        # SL/TP dari local state
        sl_p = local.get("sl_price")
        tp_p = local.get("tp_price")
        sl_pct = local.get("sl_pct", 0)
        tp_pct = local.get("tp_pct", 0)
        if sl_p:
            lines.append(f"   SL     : <code>{sl_p:.4f}</code>  (-{sl_pct:.2f}%)")
        if tp_p:
            lines.append(f"   TP     : <code>{tp_p:.4f}</code>  (+{tp_pct:.2f}%)")

        opened = local.get("opened_at", "")[:16]
        if opened:
            lines.append(f"   Dibuka : {opened} UTC")
        lines.append("")

    return "\n".join(lines)


def fmt_close_result(result: dict) -> str:
    """Format hasil close posisi."""
    if not result.get("success"):
        return f"❌ {result.get('message','Gagal close')}"

    pnl   = result.get("pnl_pct", 0)
    emoji = "🟢" if pnl >= 0 else "🔴"
    return (
        f"{emoji} <b>Posisi ditutup — {result['symbol']}</b>\n\n"
        f"Side  : {result.get('side','?')}\n"
        f"Qty   : {result.get('quantity','?')}\n"
        f"Entry : {result.get('entry_price', 0):.4f}\n"
        f"Exit  : {result.get('exit_price', 0):.4f}\n"
        f"Est PnL: <b>{pnl:+.2f}%</b>  (dengan leverage)\n"
        f"Alasan : {result.get('reason','manual')}\n"
        f"Order ID: <code>{result.get('order_id','?')}</code>"
    )


def fmt_help() -> str:
    return (
        "📖 <b>Crypto Bot — Daftar Command</b>\n"
        "──────────────────────────────\n\n"
        "🔍 <b>Analisis &amp; Trade</b>\n"
        "<code>/scan BTCUSDT</code>\n"
        "  Full pipeline: fetch → skor → history → ML adjust → AI → eksekusi order\n"
        "  Bot otomatis pasang SL &amp; TP setelah entry\n\n"
        "<code>/dryrun BTCUSDT</code>\n"
        "  Sama persis tapi TIDAK eksekusi order (aman untuk test)\n\n"
        "<code>/skor BTCUSDT</code>\n"
        "  Hanya hitung skor indikator, tanpa ML/AI (cepat ~5 detik)\n\n"
        "📦 <b>Batch</b>\n"
        "<code>/scanall</code>   — Scan semua pair di config (live)\n"
        "<code>/dryrunall</code> — Dry run semua pair di config\n\n"
        "💼 <b>Posisi</b>\n"
        "<code>/pos</code>\n"
        "  Lihat semua posisi terbuka + SL/TP yang dipasang\n\n"
        "<code>/close BTCUSDT</code>\n"
        "  Tutup posisi BTCUSDT (cancel SL/TP dulu, lalu market close)\n\n"
        "💰 <b>Akun</b>\n"
        "<code>/balance</code>  — Cek balance USDT Futures Testnet\n\n"
        "⚙️ <b>Info</b>\n"
        "<code>/pairs</code>    — Lihat pair yang dikonfigurasi di config.py\n"
        "<code>/help</code>     — Tampilkan pesan ini\n\n"
        "─────────────────────────────\n"
        "<i>SL/TP ditentukan oleh ML (ATR-based) lalu di-review AI.</i>\n"
        "<i>Minimum R/R = 1.5x. SL range 0.5–5%, TP range 0.75–15%.</i>"
    )
