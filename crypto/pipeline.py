# =============================================================
# pipeline.py — Full pipeline: train → backtest → predict → AI → order
# =============================================================

import logging
from typing import Callable

from ml.trainer     import train
from ml.backtest    import run_backtest, format_telegram as fmt_backtest
from ml.predictor   import predict
from ai.analyst     import analyze as ai_analyze
from order.executor       import execute_order
from order.paper_executor import execute_paper_order, has_paper_position
from config         import CONFIDENCE_MIN, PAPER_TRADING_MODE

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Cek posisi aktif
# ------------------------------------------------------------------

def _has_active_position(symbol: str) -> bool:
    try:
        import hashlib, hmac, time, urllib.parse, requests
        from config import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TRADE_URL, RECV_WINDOW

        params = {
            "symbol":     symbol.upper(),
            "timestamp":  int(time.time() * 1000),
            "recvWindow": RECV_WINDOW,
        }
        query = urllib.parse.urlencode(params)
        params["signature"] = hmac.new(
            BINANCE_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

        resp = requests.get(
            f"{BINANCE_TRADE_URL}/fapi/v2/positionRisk",
            params=params,
            headers={"X-MBX-APIKEY": BINANCE_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()

        for pos in resp.json():
            if float(pos.get("positionAmt", 0)) != 0:
                logger.info("[pipeline] %s active position: amt=%s entry=%s",
                            symbol, pos.get("positionAmt"), pos.get("entryPrice"))
                return True
        return False

    except Exception as e:
        logger.warning("[pipeline] Gagal cek posisi untuk %s: %s", symbol, e)
        return False


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run(
    symbol: str,
    interval: str = "30m",
    notify: Callable[[str], None] | None = None,
) -> dict:
    def _notify(msg: str):
        if notify:
            try:
                notify(msg)
            except Exception as e:
                logger.warning("notify error: %s", e)

    result = {
        "symbol":      symbol,
        "interval":    interval,
        "stage":       "start",
        "skipped":     False,
        "skip_reason": "",
        "messages":    [],
    }

    # ── 0. Skip jika sudah ada posisi aktif ──────────────────────
    active = has_paper_position(symbol) if PAPER_TRADING_MODE else _has_active_position(symbol)
    if active:
        msg = (
            f"⏭️ <b>{symbol}</b> — Skip: sudah punya posisi aktif.\n"
            f"<i>Dikelola oleh modul monitor terpisah.</i>"
        )
        _notify(msg)
        result.update({"stage": "skipped", "skipped": True,
                        "skip_reason": "active_position", "messages": [msg]})
        return result

    # ── 1. Training ───────────────────────────────────────────────
    _notify(f"⏳ <b>{symbol}</b> — Training ML ({interval})...")
    train_result = train(symbol, interval=interval)

    if not train_result["ok"]:
        msg = f"⚠️ <b>{symbol}</b> — Training gagal\n<code>{train_result['reason']}</code>"
        _notify(msg)
        result.update({"stage": "train_failed", "skipped": True,
                        "skip_reason": train_result["reason"], "messages": [msg]})
        return result

    result["train"] = train_result
    result["stage"] = "trained"

    # ── 2. Backtest ───────────────────────────────────────────────
    bt_result   = run_backtest(train_result)
    bt_messages = fmt_backtest(symbol, bt_result, train_result)
    for m in bt_messages:
        _notify(m)
    result["backtest"] = bt_result
    result["stage"]    = "backtested"
    result["messages"].extend(bt_messages)

    # ── 3. Predict ────────────────────────────────────────────────
    pred = predict(train_result)

    pred_msg = (
        f"🔮 <b>{symbol}</b> — ML Prediction\n"
        f"  Direction  : <b>{pred['direction']}</b>\n"
        f"  Confidence : <b>{pred['confidence']*100:.1f}%</b>\n"
        f"  P(Long)    : {pred['p_long']*100:.1f}%\n"
        f"  P(Short)   : {pred['p_short']*100:.1f}%\n"
        f"  P(Neutral) : {pred['p_neutral']*100:.1f}%\n"
        f"  Cur Price  : <code>{pred['current_price']}</code>\n"
        f"  Pred Price : <code>{pred['predicted_price']}</code>\n"
        f"  W.Total    : <code>{pred['weighted_total']:+.4f}</code>\n"
        f"  Scores     : " + " | ".join(f"{k}={v:+.0f}" for k, v in pred["scores"].items())
    )
    _notify(pred_msg)
    result["messages"].append(pred_msg)
    result["pred"]  = pred
    result["stage"] = "predicted"

    # ── 4. Skip check ─────────────────────────────────────────────
    if pred["skip"]:
        reason = (
            f"Confidence {pred['confidence']*100:.1f}% < {CONFIDENCE_MIN*100:.0f}%"
            if pred["confidence"] < CONFIDENCE_MIN else "Direction NEUTRAL"
        )
        skip_msg = f"⏭️ <b>{symbol}</b> — Skip: {reason}"
        _notify(skip_msg)
        result.update({"stage": "skipped", "skipped": True, "skip_reason": reason})
        result["messages"].append(skip_msg)
        return result

    # ── 5. AI Analysis ────────────────────────────────────────────
    _notify(f"🧠 <b>{symbol}</b> — Analisis AI...")
    ai_result = ai_analyze(pred, bt_result, train_result)

    result["ai"]    = ai_result
    result["stage"] = "ai_done"

    if not ai_result["ok"]:
        fail_msg = (
            f"❌ <b>{symbol}</b> — AI error\n"
            f"<code>{ai_result.get('reason_fail', 'unknown')}</code>"
        )
        _notify(fail_msg)
        result["messages"].append(fail_msg)
        return result

    ai_action = ai_result["action"]
    pos       = ai_result.get("position_detail", {})

    ai_msg = (
        f"🤖 <b>AI Decision — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  Action      : <b>{ai_action}</b>\n"
        f"  Entry       : <code>{ai_result['entry_price']}</code>\n"
        f"  Stop Loss   : <code>{ai_result['stop_loss']}</code>\n"
        f"  Take Profit : <code>{ai_result['take_profit']}</code>\n"
        f"  Leverage    : <b>{ai_result['leverage']}x</b>\n"
        f"  Qty Fraction: <code>{ai_result['qty_fraction']*100:.2f}%</code>\n"
        f"  ATR         : <code>{pos.get('atr', 0):.6f}</code>\n"
        f"  RR Ratio    : <code>{pos.get('rr_ratio', 0):.2f}</code>\n"
        f"  Kelly Edge  : <code>{pos.get('edge_pct', 0):+.2f}%</code>\n"
        f"  MC P5 DD    : <code>{pos.get('monte_carlo', {}).get('max_drawdown_p5', 0)*100:.1f}%</code>\n\n"
        f"<b>Reason:</b> {ai_result['reason']}"
    )
    _notify(ai_msg)
    result["messages"].append(ai_msg)

    # ── 6. Skip jika AI SKIP ──────────────────────────────────────
    if ai_action == "SKIP":
        skip_msg = f"⏭️ <b>{symbol}</b> — AI: SKIP"
        _notify(skip_msg)
        result.update({"stage": "skipped", "skipped": True, "skip_reason": "AI: SKIP"})
        result["messages"].append(skip_msg)
        return result

    # ── 7. Eksekusi Order — paper atau real ──────────────────────
    if PAPER_TRADING_MODE:
        _notify(f"📝 <b>{symbol}</b> — PAPER TRADE {ai_action} (simulasi, tidak ke Binance)...")
        order_result = execute_paper_order(ai_result, pred, notify_fn=notify)
    else:
        _notify(f"📤 <b>{symbol}</b> — Mengirim entry order {ai_action}...")
        order_result = execute_order(ai_result, pred, notify_fn=notify)

    result["order"] = order_result
    result["stage"] = "order_done"

    if not order_result["ok"]:
        fail_msg = (
            f"❌ <b>{symbol}</b> — Order gagal\n"
            f"<code>{order_result.get('reason_fail', 'unknown')}</code>"
        )
        _notify(fail_msg)
        result["messages"].append(fail_msg)
        return result

    side_emoji = "🟢" if order_result["side"] == "BUY" else "🔴"
    paper_tag  = " (PAPER)" if order_result.get("paper") else ""
    order_msg = (
        f"{side_emoji} <b>ENTRY ORDER SENT{paper_tag} — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  Order ID    : <code>{order_result['order_id']}</code>\n"
        f"  Side        : <b>{order_result['side']}</b>\n"
        f"  Qty         : <code>{order_result['qty']}</code>\n"
        f"  Entry       : <code>{order_result['entry_price']}</code>\n"
        f"  Stop Loss   : <code>{order_result['stop_loss']}</code>\n"
        f"  Take Profit : <code>{order_result['take_profit']}</code>\n"
        f"  Leverage    : <b>{order_result['leverage']}x</b>\n"
        f"  Qty Fraction: <code>{order_result['qty_fraction']*100:.2f}%</code>\n"
        f"  Margin Used : <code>{order_result['balance_used']} USDT</code>\n"
        f"  <i>{order_result.get('note', '')}</i>"
    )
    _notify(order_msg)
    result["messages"].append(order_msg)
    result["stage"] = "completed"

    return result
