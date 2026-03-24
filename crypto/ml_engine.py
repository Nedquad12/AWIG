"""
ml_engine.py — XGBoost: adjust weights + prediksi arah harga 3 candle ke depan

Flow:
  1. Load score history dari history_manager
  2. Buat label: close[+3] >= +0.5% → 1 (naik), <= -0.5% → -1 (turun), else 0
  3. Train XGBoost
  4. Hitung SL/TP suggestion dari ATR historis
  5. Return: weights baru + prediksi (label, confidence, proba, sl_pct, tp_pct)

SL/TP calculation:
  - ATR14 dihitung dari price history (Wilder smoothing)
  - SL = ATR * SL_ATR_MULT  (dalam %)
  - TP = SL * TP_RR_RATIO   (Risk/Reward)
  - Nilai di-clamp ke SL_MIN_PCT, SL_MAX_PCT, TP_MIN_PCT, TP_MAX_PCT dari config
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from history_manager import load_history
from weight_manager import FEATURES, DEFAULT_WEIGHTS, load_weights, save_weights
from config import (
    SL_MIN_PCT, SL_MAX_PCT, TP_MIN_PCT, TP_MAX_PCT,
    SL_ATR_MULT, TP_RR_RATIO,
)

logger = logging.getLogger(__name__)


# ── ATR-based SL/TP calculation ───────────────────────────────────────────────

def _calc_atr14(history: list[dict], period: int = 14) -> float:
    """Hitung ATR14 Wilder dari price history."""
    if len(history) < period + 1:
        return 0.0

    highs  = [float(h.get("high",  h["price"])) for h in history]
    lows   = [float(h.get("low",   h["price"])) for h in history]
    closes = [float(h["price"]) for h in history]

    tr = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i]  - closes[i-1]),
        ))

    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period

    return atr


def suggest_sl_tp(symbol: str) -> tuple[float, float]:
    """
    Hitung saran SL% dan TP% berbasis ATR14 dari history terbaru.

    Returns:
        (sl_pct, tp_pct) — keduanya positif, sudah di-clamp ke config range
    """
    history = load_history(symbol)
    if not history or len(history) < 20:
        # Fallback: gunakan default minimum
        sl = SL_MIN_PCT * 1.5
        tp = sl * TP_RR_RATIO
        return round(sl, 2), round(tp, 2)

    last_price = float(history[-1]["price"])
    if last_price <= 0:
        sl = SL_MIN_PCT * 1.5
        tp = sl * TP_RR_RATIO
        return round(sl, 2), round(tp, 2)

    atr       = _calc_atr14(history)
    atr_pct   = (atr / last_price) * 100

    sl_pct = atr_pct * SL_ATR_MULT
    tp_pct = sl_pct  * TP_RR_RATIO

    # Clamp ke range yang diizinkan
    sl_pct = max(SL_MIN_PCT, min(sl_pct, SL_MAX_PCT))
    tp_pct = max(TP_MIN_PCT, min(tp_pct, TP_MAX_PCT))

    # Pastikan R/R minimum 1.5x
    if tp_pct < sl_pct * 1.5:
        tp_pct = sl_pct * 1.5

    logger.debug(
        f"[{symbol}] ATR={atr:.4f} ({atr_pct:.2f}%) → "
        f"SL={sl_pct:.2f}% TP={tp_pct:.2f}%"
    )
    return round(sl_pct, 2), round(tp_pct, 2)

LABEL_UP_PCT   =  0.005
LABEL_DOWN_PCT = -0.005
LOOKAHEAD      =  3
MIN_BARS       =  30


def _build_df(symbol: str) -> Optional[pd.DataFrame]:
    history = load_history(symbol)
    if not history or len(history) < LOOKAHEAD + MIN_BARS:
        return None

    df = pd.DataFrame(history)
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    prices = df["price"].values
    labels = []
    for i in range(len(prices)):
        if i + LOOKAHEAD < len(prices):
            ret = (prices[i + LOOKAHEAD] - prices[i]) / prices[i]
            if ret >= LABEL_UP_PCT:
                labels.append(1)
            elif ret <= LABEL_DOWN_PCT:
                labels.append(-1)
            else:
                labels.append(0)
        else:
            labels.append(None)

    df["label"] = labels
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    return df


def run_ml_weights(symbol: str) -> dict:
    """
    Train XGBoost → update weights → return result dict.

    Returns:
        {
          "success": bool,
          "message": str,
          "weights_before": dict,
          "weights_after": dict,
          "accuracy_before": float,
          "accuracy_after": float,
          "n_bars": int,
        }
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"success": False, "message": "XGBoost belum terinstall. pip install xgboost"}

    df = _build_df(symbol)
    if df is None:
        return {"success": False, "message": f"Data tidak cukup untuk {symbol}. Jalankan /scan dulu."}

    w_before = load_weights(symbol)

    X_cols = FEATURES
    X = df[X_cols].astype(float).values
    y = df["label"].values + 1   # -1→0, 0→1, 1→2

    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(set(y_train)) < 2:
        return {"success": False, "message": "Label tidak cukup variatif untuk training."}

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0, num_class=3,
        objective="multi:softmax",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Accuracy before (dengan weight lama = semua 1.0)
    totals_before = np.array([
        sum(float(df[f].iloc[split + i]) * float(w_before.get(f, 1.0)) for f in FEATURES)
        for i in range(len(X_test))
    ])
    pred_before = np.where(totals_before >= 1.0, 2, np.where(totals_before <= -1.0, 0, 1))
    acc_before = float((pred_before == y_test).mean())

    # Feature importance → weight baru
    raw_imp  = model.feature_importances_
    mean_imp = float(np.mean(raw_imp)) or 1.0
    w_new    = {FEATURES[i]: round(float(raw_imp[i]) / mean_imp, 6) for i in range(len(FEATURES))}
    save_weights(symbol, w_new)

    # Accuracy after
    totals_after = np.array([
        sum(float(df[FEATURES[j]].iloc[split + i]) * w_new[FEATURES[j]] for j in range(len(FEATURES)))
        for i in range(len(X_test))
    ])
    pred_after = np.where(totals_after >= 1.0, 2, np.where(totals_after <= -1.0, 0, 1))
    acc_after = float((pred_after == y_test).mean())

    return {
        "success":          True,
        "message":          "OK",
        "weights_before":   w_before,
        "weights_after":    w_new,
        "accuracy_before":  round(acc_before * 100, 1),
        "accuracy_after":   round(acc_after * 100, 1),
        "n_bars":           len(df),
    }


def predict_direction(symbol: str) -> dict:
    """
    Prediksi arah harga 3 candle ke depan dari score history.

    Returns:
        {
          "label":      "NAIK" | "TURUN" | "NETRAL",
          "confidence": float (0-100),
          "proba_up":   float,
          "proba_down": float,
          "proba_flat": float,
          "win_rate":   float,
          "n_train":    int,
          "error":      str | None,
        }
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"error": "XGBoost belum terinstall"}

    df = _build_df(symbol)
    if df is None or len(df) < MIN_BARS + 5:
        return {"error": f"Data tidak cukup (min {MIN_BARS} bar)"}

    weights = load_weights(symbol)
    df["total"] = sum(
        df[f].astype(float) * float(weights.get(f, 1.0)) for f in FEATURES
    )

    X_cols = FEATURES + ["total"]
    for col in X_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[X_cols].astype(float).values
    y = df["label"].values + 1

    split = int(len(X) * 0.7)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if len(set(y_tr)) < 2:
        return {"error": "Label tidak cukup variatif"}

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0, num_class=3,
        objective="multi:softmax",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    win_rate = float((model.predict(X_te) == y_te).mean()) * 100

    proba      = model.predict_proba(X[-1].reshape(1, -1))[0]
    pred_class = int(np.argmax(proba))
    label_map  = {0: "TURUN", 1: "NETRAL", 2: "NAIK"}

    importances = {X_cols[i]: float(model.feature_importances_[i]) for i in range(len(X_cols))}
    top3 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]

    # Hitung SL/TP suggestion dari ATR
    sl_pct, tp_pct = suggest_sl_tp(symbol)

    return {
        "label":      label_map[pred_class],
        "confidence": round(float(proba[pred_class]) * 100, 1),
        "proba_up":   round(float(proba[2]) * 100, 1),
        "proba_flat": round(float(proba[1]) * 100, 1),
        "proba_down": round(float(proba[0]) * 100, 1),
        "win_rate":   round(win_rate, 1),
        "n_train":    len(X_tr),
        "top3_feat":  top3,
        "sl_pct":     sl_pct,
        "tp_pct":     tp_pct,
        "error":      None,
    }
