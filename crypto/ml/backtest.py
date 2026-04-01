"""
ml/backtest.py — Evaluasi model scoring before vs after ML weight adjustment.

Dipanggil setelah trainer.py selesai — menerima hasil train langsung
(bukan fetch ulang) untuk efisiensi.

Threshold sinyal:
  weighted_total >= +1.0  → prediksi naik
  weighted_total <= -1.0  → prediksi turun
  else                    → tidak ada sinyal
"""

import logging
import numpy as np
import pandas as pd

from ml.weight_manager import FEATURES, apply_weights

logger = logging.getLogger(__name__)

SIGNAL_UP   =  1.0
SIGNAL_DOWN = -1.0


# ------------------------------------------------------------------
# Core evaluasi
# ------------------------------------------------------------------

def _evaluate(feat_df: pd.DataFrame, weights: dict) -> dict:
    """Evaluasi metrik model dengan bobot tertentu."""
    totals = np.array([
        apply_weights({f: row[f] for f in FEATURES}, weights)
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

    tp_up = int(((pred_up)   & (labels == 1)).sum())
    tp_dn = int(((pred_down) & (labels == -1)).sum())

    prec_up    = tp_up / n_signal_up if n_signal_up > 0 else 0.0
    prec_dn    = tp_dn / n_signal_dn if n_signal_dn > 0 else 0.0
    n_sig_tot  = n_signal_up + n_signal_dn
    accuracy   = (tp_up + tp_dn) / n_sig_tot if n_sig_tot > 0 else 0.0

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
        "prec_up":      round(prec_up,  4),
        "prec_dn":      round(prec_dn,  4),
        "winrate_up":   round(prec_up,  4),
        "winrate_dn":   round(prec_dn,  4),
        "accuracy":     round(accuracy, 4),
        "score_mean":   round(float(np.mean(totals)), 4),
        "score_std":    round(float(np.std(totals)),  4),
        "score_max":    round(float(np.max(totals)),  4),
        "score_min":    round(float(np.min(totals)),  4),
    }


# ------------------------------------------------------------------
# Public: run_backtest
# ------------------------------------------------------------------

def run_backtest(train_result: dict) -> dict:
    """
    Evaluasi before vs after ML adjustment.

    Args:
        train_result: dict output dari ml/trainer.py

    Returns:
        {
          "before": dict metrik,
          "after":  dict metrik,
          "delta":  dict (accuracy, winrate_up, winrate_dn),
          "summary_text": str (untuk dikirim ke AI),
        }
    """
    feat_df        = train_result["feature_df"]
    weights_before = train_result["weights_before"]
    weights_after  = train_result["weights_after"]
    symbol         = train_result["symbol"]

    logger.info("[backtest] Evaluating %s before/after ML...", symbol)

    m_before = _evaluate(feat_df, weights_before)
    m_after  = _evaluate(feat_df, weights_after)

    delta = {
        "accuracy":   round(m_after["accuracy"]   - m_before["accuracy"],   4),
        "winrate_up": round(m_after["winrate_up"]  - m_before["winrate_up"],  4),
        "winrate_dn": round(m_after["winrate_dn"]  - m_before["winrate_dn"],  4),
    }

    # Teks ringkas untuk konteks AI
    summary_text = (
        f"Backtest {symbol} ({train_result['n_candles']} candles, {train_result['interval']}):\n"
        f"  BEFORE → Accuracy: {m_before['accuracy']*100:.1f}%, "
        f"WinRate Long: {m_before['winrate_up']*100:.1f}%, "
        f"WinRate Short: {m_before['winrate_dn']*100:.1f}%\n"
        f"  AFTER  → Accuracy: {m_after['accuracy']*100:.1f}%, "
        f"WinRate Long: {m_after['winrate_up']*100:.1f}%, "
        f"WinRate Short: {m_after['winrate_dn']*100:.1f}%\n"
        f"  Signal bars: Long {m_after['n_signal_up']}, Short {m_after['n_signal_dn']}, "
        f"No-signal {m_after['n_no_signal']}\n"
        f"  Score distribution: mean={m_after['score_mean']:+.2f}, "
        f"std={m_after['score_std']:.2f}, "
        f"max={m_after['score_max']:+.2f}, min={m_after['score_min']:+.2f}"
    )

    return {
        "before":       m_before,
        "after":        m_after,
        "delta":        delta,
        "summary_text": summary_text,
    }


# ------------------------------------------------------------------
# Format Telegram HTML
# ------------------------------------------------------------------

def format_telegram(symbol: str, bt_result: dict, train_result: dict) -> list[str]:
    """Return list[str] pesan Telegram HTML (2 pesan: summary + weight table)."""
    m_before = bt_result["before"]
    m_after  = bt_result["after"]
    d        = bt_result["delta"]
    imp      = train_result["importances"]
    w_before = train_result["weights_before"]
    w_after  = train_result["weights_after"]
    interval = train_result["interval"]
    n        = train_result["n_candles"]

    arrow = lambda v: "▲" if v > 0.001 else ("▼" if v < -0.001 else "─")

    msg1 = "\n".join([
        f"🤖 <b>ML Backtest — {symbol} {interval} ({n} candles)</b>",
        f"─────────────────────────",
        f"",
        f"<b>SEBELUM (default weight):</b>",
        f"  🎯 Accuracy   : {m_before['accuracy']*100:.1f}%",
        f"  💹 WinRate ▲  : {m_before['winrate_up']*100:.1f}%  ({m_before['n_signal_up']} sinyal)",
        f"  💹 WinRate ▼  : {m_before['winrate_dn']*100:.1f}%  ({m_before['n_signal_dn']} sinyal)",
        f"",
        f"<b>SESUDAH (ML-adjusted weight):</b>",
        f"  🎯 Accuracy   : {m_after['accuracy']*100:.1f}%",
        f"  💹 WinRate ▲  : {m_after['winrate_up']*100:.1f}%  ({m_after['n_signal_up']} sinyal)",
        f"  💹 WinRate ▼  : {m_after['winrate_dn']*100:.1f}%  ({m_after['n_signal_dn']} sinyal)",
        f"",
        f"<b>Delta:</b>",
        f"  {arrow(d['accuracy'])}  Accuracy   : {d['accuracy']*100:+.1f}%",
        f"  {arrow(d['winrate_up'])} WinRate ▲ : {d['winrate_up']*100:+.1f}%",
        f"  {arrow(d['winrate_dn'])} WinRate ▼ : {d['winrate_dn']*100:+.1f}%",
    ])

    from ml.weight_manager import FEATURES
    sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    weight_lines = [f"📐 <b>Feature Importance → Weight ({symbol})</b>", "─────────────────────────"]
    for feat, imp_val in sorted_feats:
        wo = w_before.get(feat, 1.0)
        wn = w_after.get(feat, 1.0)
        bar = "█" * min(int(imp_val * 20), 10) + "░" * max(0, 10 - int(imp_val * 20))
        weight_lines.append(
            f"  {feat:<6} imp={imp_val:.3f} [{bar}]  {wo:+.3f} → <b>{wn:+.3f}</b> {arrow(wn - wo)}"
        )
    msg2 = "\n".join(weight_lines)

    return [msg1, msg2]
