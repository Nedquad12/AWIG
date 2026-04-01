import logging
import numpy as np
import pandas as pd

from ml.weight_manager import apply_weights

FEATURES_CORE = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]

logger = logging.getLogger(__name__)

SIGNAL_UP   =  1.0
SIGNAL_DOWN = -1.0

TAKER_FEE_RATE    = 0.0004   
ROUND_TRIP_COST   = TAKER_FEE_RATE * 2   
SLIPPAGE_ESTIMATE = 0.0002   
TOTAL_COST        = ROUND_TRIP_COST + SLIPPAGE_ESTIMATE * 2   


def _evaluate(feat_df: pd.DataFrame, weights: dict, include_costs: bool = True) -> dict:

    totals = np.array([
        apply_weights({f: row[f] for f in FEATURES_CORE}, weights)
        for _, row in feat_df.iterrows()
    ])
    labels = feat_df["label"].values

    pred_up   = totals >= SIGNAL_UP
    pred_down = totals <= SIGNAL_DOWN
    pred_none = ~pred_up & ~pred_down

    n_total     = len(feat_df)
    n_signal_up = int(pred_up.sum())
    n_signal_dn = int(pred_down.sum())
    n_no_signal = int(pred_none.sum())

    if include_costs:

        COST_DISCOUNT = 0.97                       
                               
    else:
        COST_DISCOUNT = 1.0

    tp_up = int(((pred_up)   & (labels == 1)).sum())
    tp_dn = int(((pred_down) & (labels == -1)).sum())

    tp_up_adj = tp_up * COST_DISCOUNT
    tp_dn_adj = tp_dn * COST_DISCOUNT

    prec_up   = tp_up_adj / n_signal_up if n_signal_up > 0 else 0.0
    prec_dn   = tp_dn_adj / n_signal_dn if n_signal_dn > 0 else 0.0
    n_sig_tot = n_signal_up + n_signal_dn
    accuracy  = (tp_up_adj + tp_dn_adj) / n_sig_tot if n_sig_tot > 0 else 0.0

    return {
        "n_bars":       n_total,
        "n_label_up":   int((labels == 1).sum()),
        "n_label_dn":   int((labels == -1).sum()),
        "n_label_nt":   int((labels == 0).sum()),
        "n_signal_up":  n_signal_up,
        "n_signal_dn":  n_signal_dn,
        "n_no_signal":  n_no_signal,
        "tp_up":        tp_up,
        "tp_dn":        tp_dn,
        "tp_up_net":    round(tp_up_adj, 2), 
        "tp_dn_net":    round(tp_dn_adj, 2),
        "prec_up":      round(prec_up,  4),
        "prec_dn":      round(prec_dn,  4),
        "winrate_up":   round(prec_up,  4),  
        "winrate_dn":   round(prec_dn,  4),
        "accuracy":     round(accuracy, 4),
        "score_mean":   round(float(np.mean(totals)), 4),
        "score_std":    round(float(np.std(totals)),  4),
        "score_max":    round(float(np.max(totals)),  4),
        "score_min":    round(float(np.min(totals)),  4),
        "total_cost_pct": round(TOTAL_COST * 100, 4),   
    }


def run_backtest(train_result: dict) -> dict:

    feat_df        = train_result["feature_df"]
    weights_before = train_result["weights_before"]
    weights_after  = train_result["weights_after"]
    symbol         = train_result["symbol"]

    logger.info("[backtest] Evaluating %s before/after ML (with cost adjustment)...", symbol)

    m_before = _evaluate(feat_df, weights_before, include_costs=True)
    m_after  = _evaluate(feat_df, weights_after,  include_costs=True)

    m_before_gross = _evaluate(feat_df, weights_before, include_costs=False)
    m_after_gross  = _evaluate(feat_df, weights_after,  include_costs=False)

    delta = {
        "accuracy":   round(m_after["accuracy"]   - m_before["accuracy"],   4),
        "winrate_up": round(m_after["winrate_up"]  - m_before["winrate_up"],  4),
        "winrate_dn": round(m_after["winrate_dn"]  - m_before["winrate_dn"],  4),
    }

    summary_text = (
        f"Backtest {symbol} ({train_result['n_candles']} candles, {train_result['interval']}) "
        f"[NET of ~{TOTAL_COST*100:.2f}% round-trip cost]:\n"
        f"  BEFORE → Accuracy: {m_before['accuracy']*100:.1f}%, "
        f"WinRate Long: {m_before['winrate_up']*100:.1f}%, "
        f"WinRate Short: {m_before['winrate_dn']*100:.1f}%\n"
        f"  AFTER  → Accuracy: {m_after['accuracy']*100:.1f}%, "
        f"WinRate Long: {m_after['winrate_up']*100:.1f}%, "
        f"WinRate Short: {m_after['winrate_dn']*100:.1f}%\n"
        f"  GROSS (before cost) → WinRate Long: {m_after_gross['winrate_up']*100:.1f}%, "
        f"WinRate Short: {m_after_gross['winrate_dn']*100:.1f}%\n"
        f"  Signal bars: Long {m_after['n_signal_up']}, Short {m_after['n_signal_dn']}, "
        f"No-signal {m_after['n_no_signal']}\n"
        f"  Score distribution: mean={m_after['score_mean']:+.2f}, "
        f"std={m_after['score_std']:.2f}, "
        f"max={m_after['score_max']:+.2f}, min={m_after['score_min']:+.2f}"
    )

    return {
        "before":        m_before,
        "after":         m_after,
        "before_gross":  m_before_gross,
        "after_gross":   m_after_gross,
        "delta":         delta,
        "summary_text":  summary_text,
        "cost_rate":     TOTAL_COST,
    }


def format_telegram(symbol: str, bt_result: dict, train_result: dict) -> list[str]:
    """Return list[str] pesan Telegram HTML (2 pesan: summary + weight table)."""
    m_before = bt_result["before"]
    m_after  = bt_result["after"]
    m_ag     = bt_result.get("after_gross", m_after)
    d        = bt_result["delta"]
    imp      = train_result["importances"]
    w_before = train_result["weights_before"]
    w_after  = train_result["weights_after"]
    interval = train_result["interval"]
    n        = train_result["n_candles"]
    cost_pct = bt_result.get("cost_rate", TOTAL_COST) * 100

    arrow = lambda v: "▲" if v > 0.001 else ("▼" if v < -0.001 else "─")

    msg1 = "\n".join([
        f"🤖 <b>ML Backtest — {symbol} {interval} ({n} candles)</b>",
        f"💸 Est. cost per trade: ~{cost_pct:.2f}% (fee + slippage)",
        f"─────────────────────────",
        f"",
        f"<b>SEBELUM (default weight):</b>",
        f"  🎯 Accuracy   : {m_before['accuracy']*100:.1f}% <i>(net of cost)</i>",
        f"  💹 WinRate ▲  : {m_before['winrate_up']*100:.1f}%  ({m_before['n_signal_up']} sinyal)",
        f"  💹 WinRate ▼  : {m_before['winrate_dn']*100:.1f}%  ({m_before['n_signal_dn']} sinyal)",
        f"",
        f"<b>SESUDAH (ML-adjusted weight):</b>",
        f"  🎯 Accuracy   : {m_after['accuracy']*100:.1f}% <i>(net of cost)</i>",
        f"  💹 WinRate ▲  : {m_after['winrate_up']*100:.1f}%  ({m_after['n_signal_up']} sinyal)",
        f"  💹 WinRate ▼  : {m_after['winrate_dn']*100:.1f}%  ({m_after['n_signal_dn']} sinyal)",
        f"  📊 Gross ▲    : {m_ag['winrate_up']*100:.1f}%  (before cost adj.)",
        f"  📊 Gross ▼    : {m_ag['winrate_dn']*100:.1f}%  (before cost adj.)",
        f"",
        f"<b>Delta (net):</b>",
        f"  {arrow(d['accuracy'])}  Accuracy   : {d['accuracy']*100:+.1f}%",
        f"  {arrow(d['winrate_up'])} WinRate ▲ : {d['winrate_up']*100:+.1f}%",
        f"  {arrow(d['winrate_dn'])} WinRate ▼ : {d['winrate_dn']*100:+.1f}%",
    ])

    from ml.weight_manager import FEATURES_CORE
    sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    weight_lines = [f"📐 <b>Feature Importance → Weight ({symbol})</b>", "─────────────────────────"]
    for feat, imp_val in sorted_feats:
        wo = w_before.get(feat, 1.0)
        wn = w_after.get(feat, 1.0)
        bar = "█" * min(int(imp_val * 20), 10) + "░" * max(0, 10 - int(imp_val * 20))
        weight_lines.append(
            f"  {feat:<8} imp={imp_val:.3f} [{bar}]  {wo:+.3f} → <b>{wn:+.3f}</b> {arrow(wn - wo)}"
        )
    weight_lines.append("")
    weight_lines.append("<i>⚠️ Funding & LSR: bobot 1.0 (tidak di-train, real-time only)</i>")
    msg2 = "\n".join(weight_lines)

    return [msg1, msg2]
