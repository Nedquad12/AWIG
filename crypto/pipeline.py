import logging
from typing import Callable
from ml.trainer           import train
from ml.wfv               import run_wfv, format_telegram as fmt_wfv
from ml.predictor         import predict
from ai.board             import verify as board_verify, format_verdict
from order.paper_executor import execute_paper_order, has_paper_position
from config               import CONFIDENCE_MIN, PAPER_TRADING_MODE

logger = logging.getLogger(__name__)


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
        logger.warning("[pipeline] Gagal cek posisi %s: %s", symbol, e)
        return False


def run(
    symbol:         str,
    interval:       str = "4h",
    notify:         Callable[[str], None] | None = None,
    fetch_slippage: bool  = True,
    wti_pct:        float = 0.0,
    session_id:     str   = "", # untuk tracking cancel sesi di monitor
    dry_run               = False,
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

    _notify(f"⏳ <b>{symbol}</b> — Training ML ({interval})...")
    train_result = train(symbol, interval=interval)

    if not train_result["ok"]:
        msg = f"⚠️ <b>{symbol}</b> — Training gagal\n<code>{train_result['reason']}</code>"
        _notify(msg)
        result.update({"stage": "train_failed", "skipped": True,
                       "skip_reason": train_result["reason"], "messages": [msg]})
        return result

    regime      = train_result.get("regime", "Unknown")
    regime_info = train_result.get("regime_info", {})

    regime_msg = (
        f"🌡️ <b>{symbol}</b> — Regime: <b>{regime}</b>\n"
        f"  ADX     : {regime_info.get('adx', 0):.1f}\n"
        f"  ATR pct : {regime_info.get('atr_pct', 0):.0f}%\n"
        f"  Autocorr: {regime_info.get('autocorr', 0):+.3f}\n"
        f"  <i>{regime_info.get('description', '')}</i>"
    )
    _notify(regime_msg)
    result["train"] = train_result
    result["stage"] = "trained"
    result["messages"].append(regime_msg)

    _notify(f"📊 <b>{symbol}</b> — Walk-Forward Validation...")
    wfv_result   = run_wfv(train_result, fetch_slippage=fetch_slippage)
    wfv_messages = fmt_wfv(symbol, wfv_result, train_result)
    for m in wfv_messages:
        _notify(m)

    result["wfv"]   = wfv_result
    result["stage"] = "wfv_done"
    result["messages"].extend(wfv_messages)

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

    # ── Board verification (replace DeepSeek) ───────────────────────────────
    # Compute posisi long dan short untuk board
    from ml.kelly  import compute_position
    from ml.wfv    import TRAIN_END_IDX
    from config    import RISK_PER_TRADE_PCT

    wfv_after   = wfv_result.get("after", {})
    kelly_mult  = train_result.get("kelly_multiplier", 0.20)
    risk_max    = RISK_PER_TRADE_PCT / 100

    # Blend winrate (sama seperti analyst.py dulu)
    candle_result = train_result.get("candle_result")
    candle_bt     = candle_result.get("backtest", {}) if candle_result and candle_result.get("ok") else {}
    wr_candle_up  = candle_bt.get("winrate_up") if candle_bt else None
    wr_candle_dn  = candle_bt.get("winrate_dn") if candle_bt else None

    def _blend(wr_ind, wr_can):
        if wr_can is None or wr_can <= 0:
            return wr_ind
        return round((wr_ind * 1.0 + wr_can * 3.5) / 4.5, 4)

    wr_long  = _blend(float(wfv_after.get("winrate_up", 0.0)), wr_candle_up)
    wr_short = _blend(float(wfv_after.get("winrate_dn", 0.0)), wr_candle_dn)
    n_sig_up = wfv_after.get("n_signal_up", 0) + candle_bt.get("n_signal_up", 0)
    n_sig_dn = wfv_after.get("n_signal_dn", 0) + candle_bt.get("n_signal_dn", 0)

    # Fetch max leverage Binance untuk symbol ini agar kelly tidak output
    # leverage yang melebihi batas Binance (akan ditolak saat set leverage)
    try:
        from order.executor import get_max_leverage as _get_max_lev
        _max_lev_binance = _get_max_lev(symbol)
    except Exception as _e:
        logger.warning("[pipeline] Gagal fetch max leverage %s: %s — pakai default", symbol, _e)
        from ml.kelly import MAX_LEVERAGE as _max_lev_binance

    pos_long = compute_position(
        df=train_result["raw_df"], direction="LONG",
        winrate=wr_long, n_signals=n_sig_up,
        risk_per_trade=risk_max, max_fraction=risk_max,
        train_end=TRAIN_END_IDX, kelly_multiplier_override=kelly_mult,
        max_leverage_binance=_max_lev_binance,
    )
    pos_short = compute_position(
        df=train_result["raw_df"], direction="SHORT",
        winrate=wr_short, n_signals=n_sig_dn,
        risk_per_trade=risk_max, max_fraction=risk_max,
        train_end=TRAIN_END_IDX, kelly_multiplier_override=kelly_mult,
        max_leverage_binance=_max_lev_binance,
    )

    _notify(f"🔲 <b>{symbol}</b> — Board verification...")
    board_action, board_reason = board_verify(pred, wfv_result, train_result, pos_long, pos_short)

    result["stage"] = "board_done"

    # Pilih posisi sesuai action
    pos = pos_long if board_action == "BUYING" else pos_short

    # Inject wti_pct
    board_result = {
        "ok":              board_action != "SKIP",
        "action":          board_action,
        "reason":          board_reason,
        "entry_price":     pos["entry_price"],
        "stop_loss":       pos["stop_loss"],
        "take_profit":     pos["take_profit"],
        "leverage":        pos["leverage"],
        "qty_fraction":    pos["qty_fraction"],
        "position_detail": pos,
        "regime":          regime,
        "wti_pct":         wti_pct,
    }

    board_msg = format_verdict(board_action, board_reason, pos)
    _notify(board_msg)
    result["messages"].append(board_msg)
    result["board"] = board_result

    if board_action == "SKIP":
        result.update({"stage": "skipped", "skipped": True, "skip_reason": board_reason})
        return result

    ai_result = board_result  # alias agar kode di bawah tidak perlu diubah
    ai_result["session_id"] = session_id  # forward ke paper_executor untuk tracking sesi

    # ── Dry-run mode: return hasil board tanpa eksekusi order ─────────────────
    # Scheduler akan memutuskan apakah ini special coin (market) atau limit order
    if dry_run:
        result["stage"]        = "board_done_dry"
        result["dry_run"]      = True
        result["board"]        = board_result
        result["pred"]         = pred
        result["wfv_result"]   = wfv_result
        result["train_result"] = train_result
        result["pos_long"]     = pos_long
        result["pos_short"]    = pos_short
        logger.info("[pipeline] dry_run=True — return tanpa order: %s %s", symbol, board_action)
        return result

    if PAPER_TRADING_MODE:
        _notify(f"📝 <b>{symbol}</b> — PAPER LIMIT ORDER {board_action}...")
        order_result = execute_paper_order(ai_result, pred, notify_fn=notify)
    else:
        _notify(f"📤 <b>{symbol}</b> — Mengirim entry order {board_action}...")
        order_result = execute_paper_order(ai_result, pred, notify_fn=notify)  # live juga via paper_executor

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
        f"{side_emoji} <b>ORDER{paper_tag} — {symbol}</b>\n"
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
        f"  <i>{order_result.get('note', '')}</i>"
    )
    _notify(order_msg)
    result["messages"].append(order_msg)
    result["stage"] = "completed"

    return result
