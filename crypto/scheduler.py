"""
scheduler.py — Auto Trading Session dengan Special Coin (2x Board) logic

PERUBAHAN UTAMA pada fungsi run_session():
  Lama : setiap koin langsung execute order setelah board approve
  Baru :
    Phase 1 — Pipeline dry_run semua kandidat, kumpulkan board_approved list
    Phase 2 — Evaluasi apakah ada "special coin" (2x lipat semua threshold board)
              Jika ada special → market order 1 koin (skor tertinggi) + max 1 limit order sisa
              Jika tidak ada   → lanjut seperti biasa max 2 limit order
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

MAX_OPEN_POSITIONS     = 10
MAX_ORDERS_PER_SCAN    = 2     # max order dibuka per sesi (normal)
MAX_FILLED_PER_SESSION = 2     # max order filled per sesi sebelum sisa di-cancel
FLOATING_DD_WARN_PCT   = 0.10

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
        "PAPER LIMIT", "PAPER MARKET ORDER", "LIMIT FILLED", "LIMIT CANCELLED",
        "Order Expired", "CLOSED", "Breakeven", "Trailing SL", "Partial TP",
        "Monitor aktif —", "Monitor Bot Online",
    ]
    if any(k in msg for k in order_keywords):
        return TOPIC_ORDERS

    # Board — verdict akhir pipeline (BUYING/SELLING/SKIP)
    board_keywords = [
        "🔲", "Board", "BUYING", "SELLING", "SPECIAL COIN",
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
        return
    await asyncio.sleep(1)
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
    def sync_notify(msg: str) -> None:
        try:
            if event_loop.is_closed():
                return
            tid = topic_id if topic_id is not None else _topic_for(msg)
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


def _check_candidate_eligibility(candidate: dict, session_id: str, bot: Bot, loop) -> tuple[bool, float]:
    """
    Cek apakah kandidat lolos filter awal (urgent CB, ban, WTI).
    Return (eligible: bool, wti_pct: float).
    """
    symbol = candidate["symbol"]

    ucb_active, ucb_banned_side = get_urgent_cb_ban()
    if ucb_active:
        candidate_dir = candidate.get("direction", "")
        if candidate_dir == ucb_banned_side:
            logger.info("[scheduler] %s skip — Urgent CB ban %s", symbol, ucb_banned_side)
            asyncio.run_coroutine_threadsafe(
                _broadcast(
                    bot,
                    f"🚨 <b>{symbol}</b> skip — Urgent CB: <b>{ucb_banned_side} di-ban 1 sesi</b>",
                    TOPIC_CB,
                ),
                loop,
            )
            return False, 0.0

    banned, ban_rem = is_banned(symbol)
    if banned:
        logger.info("[scheduler] %s banned %d sesi — skip", symbol, ban_rem)
        return False, 0.0

    wti_pct = 0.0
    try:
        from wti_crypto import get_wti
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut     = ex.submit(get_wti, symbol)
            wti_res = fut.result(timeout=10)
        if wti_res:
            wti_pct = float(wti_res.get("wti_pct", 0.0))
    except Exception as e:
        logger.warning("[scheduler] WTI %s error: %s", symbol, e)

    wti_ok, wti_reason = check_wti_slot(wti_pct)
    if not wti_ok:
        logger.info("[scheduler] %s WTI filter: %s", symbol, wti_reason)
        return False, 0.0

    return True, wti_pct


async def run_session(bot: Bot) -> None:
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    loop       = asyncio.get_event_loop()

    logger.info("[scheduler] ===== SESSION %s START =====", session_id)

    risk_state = tick_session()

    # ── Circuit Breaker ───────────────────────────────────────────────────────
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

    # ── BTC Volume Spike Cooldown ─────────────────────────────────────────────
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

    # ── Posisi penuh ──────────────────────────────────────────────────────────
    active_count = _count_active_positions()
    if active_count >= MAX_OPEN_POSITIONS:
        logger.info("[scheduler] Sesi %s di-skip — posisi penuh %d/%d", session_id,
                    active_count, MAX_OPEN_POSITIONS)
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

    # ── Scan ──────────────────────────────────────────────────────────────────
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

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Jalankan pipeline DRY-RUN semua kandidat, kumpulkan board results
    # ═══════════════════════════════════════════════════════════════════════════

    board_approved = []   # list of (symbol, result, wti_pct) yang lolos board

    for candidate in passed:
        active_now = _count_active_positions()
        if active_now >= MAX_OPEN_POSITIONS:
            logger.info("[scheduler] Posisi penuh saat phase 1 — stop scan")
            break

        if loop.is_closed():
            return

        symbol = candidate["symbol"]

        # Filter awal (CB, ban, WTI)
        eligible, wti_pct = await loop.run_in_executor(
            None,
            lambda c=candidate: _check_candidate_eligibility(c, session_id, bot, loop),
        )
        if not eligible:
            continue

        logger.info("[scheduler] Dry-run pipeline %s (wti=%.1f%%)", symbol, wti_pct)

        try:
            result = await loop.run_in_executor(
                None,
                lambda s=symbol, w=wti_pct: run_pipeline(
                    s,
                    interval    = DEFAULT_INTERVAL,
                    notify      = _make_notify(loop, bot),
                    wti_pct     = w,
                    session_id  = session_id,
                    dry_run     = True,           # ← kunci: tidak langsung execute
                ),
            )
        except GeneratorExit:
            return
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("[scheduler] Dry-run pipeline error %s: %s", symbol, e)
            if not loop.is_closed():
                await _broadcast(
                    bot,
                    f"❌ <b>Pipeline error {symbol}:</b> <code>{e}</code>",
                    topic_id=TOPIC_ERROR,
                )
            continue

        # Kumpulkan yang board approve
        board = result.get("board", {})
        if board.get("ok") and result.get("stage") == "board_done_dry":
            board_approved.append({
                "symbol":    symbol,
                "result":    result,
                "wti_pct":   wti_pct,
            })
            logger.info("[scheduler] Board approved (dry): %s", symbol)

    if not board_approved:
        logger.info("[scheduler] Sesi %s — tidak ada koin lolos board", session_id)
        _try_save_daily(bot, loop)
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Evaluasi special coin (2x board) dan tentukan alokasi order
    # ═══════════════════════════════════════════════════════════════════════════

    from ai.board import compute_special_score, format_special_verdict

    # Hitung special score untuk semua kandidat yang lolos board
    scored_candidates = []
    for entry in board_approved:
        result = entry["result"]
        pred         = result.get("pred", {})
        wfv_result   = result.get("wfv_result", {})
        train_result = result.get("train_result", {})
        pos_long     = result.get("pos_long", {})
        pos_short    = result.get("pos_short", {})

        is_special, score, detail = compute_special_score(
            pred, wfv_result, train_result, pos_long, pos_short,
        )

        scored_candidates.append({
            **entry,
            "is_special":  is_special,
            "score":       score,
            "detail":      detail,
        })

    # Pisahkan special dan normal
    specials = [c for c in scored_candidates if c["is_special"]]
    normals  = [c for c in scored_candidates if not c["is_special"]]

    # Urutkan special by score (tertinggi dulu)
    specials.sort(key=lambda x: x["score"], reverse=True)
    # Urutkan normal by score juga (untuk pilih yang terbaik jika perlu cancel)
    normals.sort(key=lambda x: x["score"], reverse=True)

    # ── Tentukan alokasi slot ─────────────────────────────────────────────────
    # Normal: max 2 limit order
    # Jika ada special: 1 market order (special terbaik) + max 1 limit order
    # Yang tidak masuk slot → di-skip (tidak di-order, bukan di-cancel karena belum ada order)

    selected_special = None
    selected_limits  = []

    if specials:
        selected_special = specials[0]          # ambil skor tertinggi
        # Limit order hanya 1 dari sisa (normal dulu, lalu special ke-2+ jika ada)
        remaining_limit_candidates = normals + specials[1:]
        remaining_limit_candidates.sort(key=lambda x: x["score"], reverse=True)
        selected_limits = remaining_limit_candidates[:1]  # max 1 limit order
    else:
        # Tidak ada special → max 2 limit order seperti biasa
        selected_limits = normals[:slots_this_scan]

    skipped_candidates = [
        c for c in scored_candidates
        if c is not selected_special and c not in selected_limits
    ]

    # Notif ringkasan phase 2
    special_sym  = selected_special["symbol"] if selected_special else "–"
    limit_syms   = ", ".join(c["symbol"] for c in selected_limits) or "–"
    skipped_syms = ", ".join(c["symbol"] for c in skipped_candidates) or "–"

    await _broadcast(
        bot,
        f"🎯 <b>Sesi #{session_id} — Alokasi Order</b>\n"
        f"  Board lolos   : <b>{len(board_approved)}</b> koin\n"
        f"  Special (2x)  : <b>{special_sym}</b>{'  ⭐ market order' if selected_special else ''}\n"
        f"  Limit order   : <b>{limit_syms}</b>\n"
        f"  Di-skip       : <i>{skipped_syms}</i>",
        topic_id=TOPIC_BOARD,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Eksekusi order
    # ═══════════════════════════════════════════════════════════════════════════

    from config import PAPER_TRADING_MODE
    orders_placed = 0

    # ── 3a: Special coin → market order ──────────────────────────────────────
    if selected_special:
        sym     = selected_special["symbol"]
        result  = selected_special["result"]
        wti_pct = selected_special["wti_pct"]
        detail  = selected_special["detail"]
        score   = selected_special["score"]

        board       = result["board"]
        pred        = result["pred"]
        board_action = board["action"]

        # Pilih posisi (long atau short)
        pos = result["pos_long"] if board_action == "BUYING" else result["pos_short"]

        # Notif special coin
        await _broadcast(
            bot,
            format_special_verdict(sym, score, detail, pos),
            topic_id=TOPIC_BOARD,
        )

        ai_result = {**board, "session_id": session_id, "wti_pct": wti_pct}

        if loop.is_closed():
            return

        try:
            if PAPER_TRADING_MODE:
                from order.paper_executor import execute_paper_market_order
                order_result = await loop.run_in_executor(
                    None,
                    lambda: execute_paper_market_order(
                        ai_result, pred,
                        notify_fn=_make_notify(loop, bot),
                    ),
                )
            else:
                # Live: gunakan MARKET order langsung via executor
                from order.executor import execute_market_order
                order_result = await loop.run_in_executor(
                    None,
                    lambda: execute_market_order(
                        ai_result, pred,
                        notify_fn=_make_notify(loop, bot),
                    ),
                )

            if order_result.get("ok"):
                orders_placed += 1
                logger.info("[scheduler] Special market order placed: %s", sym)
            else:
                logger.warning(
                    "[scheduler] Special market order gagal %s: %s",
                    sym, order_result.get("reason_fail"),
                )
                await _broadcast(
                    bot,
                    f"❌ <b>{sym}</b> — Market order gagal\n"
                    f"<code>{order_result.get('reason_fail', 'unknown')}</code>",
                    topic_id=TOPIC_ERROR,
                )
        except Exception as e:
            logger.exception("[scheduler] Market order error %s: %s", sym, e)
            await _broadcast(
                bot,
                f"❌ <b>Market order error {sym}:</b> <code>{e}</code>",
                topic_id=TOPIC_ERROR,
            )

    # ── 3b: Limit order untuk sisa slot ──────────────────────────────────────
    for entry in selected_limits:
        active_now = _count_active_positions()
        if active_now >= MAX_OPEN_POSITIONS:
            logger.info("[scheduler] Posisi penuh saat limit order phase — stop")
            break

        if loop.is_closed():
            return

        sym     = entry["symbol"]
        result  = entry["result"]
        wti_pct = entry["wti_pct"]
        board   = result["board"]
        pred    = result["pred"]

        ai_result    = {**board, "session_id": session_id, "wti_pct": wti_pct}
        board_action = board["action"]

        try:
            if PAPER_TRADING_MODE:
                from order.paper_executor import execute_paper_order
                _notify_fn = _make_notify(loop, bot)
                _notify_fn(f"📝 <b>{sym}</b> — PAPER LIMIT ORDER {board_action}...")
                order_result = await loop.run_in_executor(
                    None,
                    lambda s=sym, ai=ai_result, p=pred: execute_paper_order(
                        ai, p, notify_fn=_make_notify(loop, bot)
                    ),
                )
            else:
                from order.executor import execute_order
                _make_notify(loop, bot)(f"📤 <b>{sym}</b> — Mengirim entry order {board_action}...")
                order_result = await loop.run_in_executor(
                    None,
                    lambda s=sym, ai=ai_result, p=pred: execute_order(
                        ai, p, notify_fn=_make_notify(loop, bot)
                    ),
                )

            if order_result.get("ok"):
                orders_placed += 1
                side_emoji = "🟢" if order_result["side"] == "BUY" else "🔴"
                paper_tag  = " (PAPER)" if order_result.get("paper") else ""
                regime     = board.get("regime", "?")
                await _broadcast(
                    bot,
                    f"{side_emoji} <b>ORDER{paper_tag} — {sym}</b>\n"
                    f"─────────────────────────\n"
                    f"  ID          : <code>{order_result['order_id']}</code>\n"
                    f"  Side        : <b>{order_result['side']}</b>\n"
                    f"  Regime      : <b>{regime}</b>\n"
                    f"  Qty         : <code>{order_result['qty']}</code>\n"
                    f"  Entry       : <code>{order_result['entry_price']}</code>\n"
                    f"  Stop Loss   : <code>{order_result['stop_loss']}</code>\n"
                    f"  Take Profit : <code>{order_result['take_profit']}</code>\n"
                    f"  Leverage    : <b>{order_result['leverage']}x</b>\n"
                    f"  WTI         : <code>{wti_pct:.1f}%</code>\n"
                    f"  Margin Used : <code>{order_result['balance_used']} USDT</code>\n"
                    f"  <i>{order_result.get('note', '')}</i>",
                    topic_id=TOPIC_ORDERS,
                )
            else:
                await _broadcast(
                    bot,
                    f"❌ <b>{sym}</b> — Order gagal\n"
                    f"<code>{order_result.get('reason_fail', 'unknown')}</code>",
                    topic_id=TOPIC_ERROR,
                )

        except GeneratorExit:
            return
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception("[scheduler] Limit order error %s: %s", sym, e)
            if not loop.is_closed():
                await _broadcast(
                    bot,
                    f"❌ <b>Pipeline error {sym}:</b> <code>{e}</code>",
                    topic_id=TOPIC_ERROR,
                )

    if loop.is_closed():
        return

    final_active = _count_active_positions()
    logger.info(
        "[scheduler] SESSION %s END special=%s orders=%d active=%d",
        session_id,
        selected_special["symbol"] if selected_special else "none",
        orders_placed,
        final_active,
    )
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
