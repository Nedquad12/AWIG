# =============================================================
# pipeline.py — Full pipeline: train → backtest → predict → AI → order
# =============================================================

import logging
from typing import Callable

from ml.trainer    import train
from ml.backtest   import run_backtest, format_telegram as fmt_backtest
from ml.predictor  import predict
from ai.analyst    import analyze as ai_analyze
from order.executor import execute_order
from config        import CONFIDENCE_MIN

logger = logging.getLogger(__name__)


def run(
    symbol: str,
    interval: str = "30m",
    notify: Callable[[str], None] | None = None,
) -> dict:
    """
    Jalankan full pipeline untuk satu symbol.

    Returns:
        dict ringkasan hasil pipeline dengan key:
        symbol, interval, stage, skipped, skip_reason,
        train, backtest, pred, ai, order, messages
    """
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

    # ── 1. Training ───────────────────────────────────────────────
    _notify(f"⏳ <b>{symbol}</b> — Training ML ({interval})...")
    train_result = train(symbol, interval=interval)

    if not train_result["ok"]:
        msg = (
            f"⚠️ <b>{symbol}</b> — Training gagal\n"
            f"<code>{train_result['reason']}</code>"
        )
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
        f"  Scores     : " + " | ".join(
            f"{k}={v:+.0f}" for k, v in pred["scores"].items()
        )
    )
    _notify(pred_msg)
    result["messages"].append(pred_msg)
    result["pred"]  = pred
    result["stage"] = "predicted"

    # ── 4. Skip check ─────────────────────────────────────────────
    if pred["skip"]:
        reason = (
            f"Confidence {pred['confidence']*100:.1f}% < {CONFIDENCE_MIN*100:.0f}%"
            if pred["confidence"] < CONFIDENCE_MIN
            else "Direction NEUTRAL"
        )
        skip_msg = f"⏭️ <b>{symbol}</b> — Skip: {reason}"
        _notify(skip_msg)
        result.update({"stage": "skipped", "skipped": True, "skip_reason": reason})
        result["messages"].append(skip_msg)
        return result

    # ── 5. AI Analysis ────────────────────────────────────────────
    _notify(f"🧠 <b>{symbol}</b> — Mengirim data ke DeepSeek R1...")
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
    ai_msg = (
        f"🤖 <b>DeepSeek R1 — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  Action      : <b>{ai_action}</b>\n"
        f"  Entry Price : <code>{ai_result['entry_price']}</code>\n"
        f"  Stop Loss   : <code>{ai_result['stop_loss']}</code>\n"
        f"  Take Profit : <code>{ai_result['take_profit']}</code>\n"
        f"  Leverage    : <b>{ai_result['leverage']}x</b>\n\n"
        f"<b>Alasan AI:</b>\n{ai_result['reason']}"
    )
    _notify(ai_msg)
    result["messages"].append(ai_msg)

    # ── 6. Skip jika HOLD ─────────────────────────────────────────
    if ai_action == "HOLD":
        hold_msg = f"⏭️ <b>{symbol}</b> — AI memutuskan HOLD, tidak ada order."
        _notify(hold_msg)
        result.update({"stage": "hold", "skipped": True, "skip_reason": "AI: HOLD"})
        result["messages"].append(hold_msg)
        return result

    # ── 7. Eksekusi Order ─────────────────────────────────────────
    _notify(f"📤 <b>{symbol}</b> — Mengeksekusi {ai_action}...")
    order_result = execute_order(ai_result, pred)

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
    order_msg = (
        f"{side_emoji} <b>ORDER PLACED — {symbol}</b>\n"
        f"─────────────────────────\n"
        f"  Order ID    : <code>{order_result['order_id']}</code>\n"
        f"  Side        : <b>{order_result['side']}</b>\n"
        f"  Qty         : <code>{order_result['qty']}</code>\n"
        f"  Entry       : <code>{order_result['entry_price']}</code>\n"
        f"  Stop Loss   : <code>{order_result['stop_loss']}</code>\n"
        f"  Take Profit : <code>{order_result['take_profit']}</code>\n"
        f"  Leverage    : <b>{order_result['leverage']}x</b>\n"
        f"  Margin Used : <code>{order_result['balance_used']} USDT</code>"
    )
    _notify(order_msg)
    result["messages"].append(order_msg)
    result["stage"] = "completed"

    return result
