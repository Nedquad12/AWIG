import logging
import numpy as np
import pandas as pd

from ml.weight_manager import apply_weights

FEATURES_CORE = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]

logger = logging.getLogger(__name__)

SIGNAL_UP   =  1.0
SIGNAL_DOWN = -1.0

# ── Train/Test split ─────────────────────────────────────────────────
# Train : candle index 0   s/d TRAIN_END_IDX  (dilakukan di trainer.py)
# Test  : candle index TRAIN_END_IDX s/d akhir (out-of-sample, dipakai di sini)
# Default: 850 candle untuk train, sisanya (~150) untuk test
TRAIN_END_IDX  = 850
MIN_TEST_BARS  = 30   # minimum bar test agar winrate statistis bermakna


def _evaluate(feat_df: pd.DataFrame, weights: dict) -> dict:
    """
    Evaluasi murni — tanpa cost adjustment.
    feat_df harus sudah berupa test set (out-of-sample) sebelum dipanggil.
    """
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

    tp_up = int(((pred_up)   & (labels == 1)).sum())
    tp_dn = int(((pred_down) & (labels == -1)).sum())

    prec_up   = tp_up / n_signal_up if n_signal_up > 0 else 0.0
    prec_dn   = tp_dn / n_signal_dn if n_signal_dn > 0 else 0.0
    n_sig_tot = n_signal_up + n_signal_dn
    accuracy  = (tp_up + tp_dn) / n_sig_tot if n_sig_tot > 0 else 0.0

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
        "winrate_up":   round(prec_up,  4),
        "winrate_dn":   round(prec_dn,  4),
        "accuracy":     round(accuracy, 4),
        "score_mean":   round(float(np.mean(totals)), 4),
        "score_std":    round(float(np.std(totals)),  4),
        "score_max":    round(float(np.max(totals)),  4),
        "score_min":    round(float(np.min(totals)),  4),
    }


def run_backtest(train_result: dict) -> dict:
    feat_df        = train_result["feature_df"]
    weights_before = train_result["weights_before"]
    weights_after  = train_result["weights_after"]
    symbol         = train_result["symbol"]
    n_total        = len(feat_df)

    # ── Out-of-sample split ──────────────────────────────────────────
    # Gunakan candle 851–akhir sebagai test set.
    # Kalau total candle tidak cukup untuk split di 850, fallback ke 80/20.
    if n_total >= TRAIN_END_IDX + MIN_TEST_BARS:
        test_df    = feat_df.iloc[TRAIN_END_IDX:].reset_index(drop=True)
        train_used = TRAIN_END_IDX
        split_note = f"candle {TRAIN_END_IDX}–{n_total} ({len(test_df)} bars, OOS)"
    else:
        split_at   = max(int(n_total * 0.80), n_total - MIN_TEST_BARS)
        test_df    = feat_df.iloc[split_at:].reset_index(drop=True)
        train_used = split_at
        split_note = f"candle {split_at}–{n_total} ({len(test_df)} bars, fallback 80/20)"
        logger.warning(
            "[backtest] %s: total candle %d < %d, pakai fallback split di candle %d",
            symbol, n_total, TRAIN_END_IDX + MIN_TEST_BARS, split_at,
        )

    logger.info("[backtest] %s — OOS test: %s", symbol, split_note)

    if len(test_df) < MIN_TEST_BARS:
        logger.warning(
            "[backtest] %s: test set hanya %d bars — winrate tidak reliable",
            symbol, len(test_df),
        )

    m_before = _evaluate(test_df, weights_before)
    m_after  = _evaluate(test_df, weights_after)

    delta = {
        "accuracy":   round(m_after["accuracy"]   - m_before["accuracy"],   4),
        "winrate_up": round(m_after["winrate_up"]  - m_before["winrate_up"],  4),
        "winrate_dn": round(m_after["winrate_dn"]  - m_before["winrate_dn"],  4),
    }

    summary_text = (
        f"Backtest {symbol} — OUT-OF-SAMPLE | {split_note} | {train_result['interval']}\n"
        f"  Train: 0–{train_used} candle | Test: {split_note}\n"
        f"  BEFORE → Accuracy: {m_before['accuracy']*100:.1f}%, "
        f"WinRate Long: {m_before['winrate_up']*100:.1f}%, "
        f"WinRate Short: {m_before['winrate_dn']*100:.1f}%\n"
        f"  AFTER  → Accuracy: {m_after['accuracy']*100:.1f}%, "
        f"WinRate Long: {m_after['winrate_up']*100:.1f}%, "
        f"WinRate Short: {m_after['winrate_dn']*100:.1f}%\n"
        f"  Signal bars: Long {m_after['n_signal_up']}, Short {m_after['n_signal_dn']}, "
        f"No-signal {m_after['n_no_signal']}\n"
        f"  Score: mean={m_after['score_mean']:+.2f} std={m_after['score_std']:.2f} "
        f"max={m_after['score_max']:+.2f} min={m_after['score_min']:+.2f}"
    )

    return {
        "before":       m_before,
        "after":        m_after,
        "delta":        delta,
        "summary_text": summary_text,
        "test_n_bars":  len(test_df),
        "train_n_bars": train_used,
        "split_note":   split_note,
    }


def format_telegram(symbol: str, bt_result: dict, train_result: dict) -> list[str]:
    m_before = bt_result["before"]
    m_after  = bt_result["after"]
    d        = bt_result["delta"]
    imp      = train_result["importances"]
    w_before = train_result["weights_before"]
    w_after  = train_result["weights_after"]
    interval = train_result["interval"]
    n        = train_result["n_candles"]
    split    = bt_result.get("split_note", "out-of-sample")
    test_n   = bt_result.get("test_n_bars", "?")

    arrow = lambda v: "▲" if v > 0.001 else ("▼" if v < -0.001 else "─")

    msg1 = "\n".join([
        f"🤖 <b>ML Backtest — {symbol} {interval} ({n} candles)</b>",
        f"📊 <b>OUT-OF-SAMPLE</b>: {split}",
        f"   Test bars: {test_n} | Train: 0–{bt_result.get('train_n_bars','?')}",
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
        f"<b>Delta (OOS):</b>",
        f"  {arrow(d['accuracy'])}  Accuracy   : {d['accuracy']*100:+.1f}%",
        f"  {arrow(d['winrate_up'])} WinRate ▲ : {d['winrate_up']*100:+.1f}%",
        f"  {arrow(d['winrate_dn'])} WinRate ▼ : {d['winrate_dn']*100:+.1f}%",
        f"",
        f"<i>✅ Winrate dari out-of-sample test — tidak melihat training data.</i>",
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
