"""
ml/trainer.py — XGBoost trainer untuk adjust bobot indikator.

Flow:
  1. Fetch 1000 candle (30m default)
  2. Hitung skor semua indikator per candle
  3. Buat label: close[+3] / close[0] - 1 >= +0.5% → 1, <= -0.5% → -1, else 0
  4. Train XGBoost
  5. Konversi feature importance → bobot baru
  6. Simpan ke weights/<TICKER>.json

Return dict hasil training untuk dipakai backtest & predictor.
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CANDLE_LIMIT, DEFAULT_INTERVAL,
    LABEL_UP_PCT, LABEL_DOWN_PCT,
    LOOKAHEAD, MIN_CANDLE_TRAIN,
)
from indicators.binance_fetcher import get_df
from indicators import (
    score_vsa, score_fsa, score_vfa,
    score_rsi, score_macd, score_ma, score_wcc,
)
from ml.weight_manager import (
    DEFAULT_WEIGHTS, FEATURES,
    load_weights, save_weights,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Hitung skor semua indikator per-candle (rolling window)
# ------------------------------------------------------------------

def _score_at(df: pd.DataFrame, i: int) -> dict[str, float]:
    """
    Hitung skor semua indikator menggunakan data df[:i+1].
    Minimal window = 210 candle (kebutuhan MA200).
    Return dict {indicator: score} atau semua 0 jika data kurang.
    """
    window = df.iloc[:i + 1]
    if len(window) < 210:
        return {f: 0.0 for f in FEATURES}

    return {
        "vsa":  float(score_vsa(window)),
        "fsa":  float(score_fsa(window)),
        "vfa":  float(score_vfa(window)),
        "rsi":  float(score_rsi(window)),
        "macd": float(score_macd(window)),
        "ma":   float(score_ma(window)),
        "wcc":  float(score_wcc(window)),
    }


def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun feature matrix: tiap baris = skor indikator pada candle ke-i.
    Hanya baris dengan window >= 210 yang dihitung.
    """
    rows = []
    prices = df["close"].values

    for i in range(len(df)):
        # Label: return 3 candle ke depan
        if i + LOOKAHEAD >= len(df):
            label = None
        else:
            ret = (prices[i + LOOKAHEAD] - prices[i]) / prices[i]
            if ret >= LABEL_UP_PCT:
                label = 1
            elif ret <= LABEL_DOWN_PCT:
                label = -1
            else:
                label = 0

        scores = _score_at(df, i)
        scores["label"] = label
        scores["price"] = float(prices[i])
        rows.append(scores)

    result = pd.DataFrame(rows)
    # Drop baris tanpa label dan window terlalu kecil
    result = result[result["label"].notna()].copy()
    result = result[result[FEATURES].any(axis=1)].copy()  # drop semua-nol
    result["label"] = result["label"].astype(int)
    return result.reset_index(drop=True)


# ------------------------------------------------------------------
# Public: train
# ------------------------------------------------------------------

def train(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = CANDLE_LIMIT,
) -> dict:
    """
    Fetch data, train XGBoost, simpan bobot baru.

    Returns:
        {
          "ok": bool,
          "reason": str (jika ok=False),
          "symbol": str,
          "interval": str,
          "n_candles": int,
          "n_train": int,
          "n_test": int,
          "importances": dict,
          "weights_before": dict,
          "weights_after": dict,
          "feature_df": pd.DataFrame,   ← untuk backtest & predictor
          "raw_df": pd.DataFrame,        ← kline asli
        }
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"ok": False, "reason": "XGBoost belum terinstall. Jalankan: pip install xgboost"}

    symbol = symbol.upper()

    # -- Fetch data --
    logger.info("[trainer] Fetch %s candle %s %s", limit, symbol, interval)
    raw_df = get_df(symbol, interval=interval, limit=limit)
    n_candles = len(raw_df)

    if n_candles < MIN_CANDLE_TRAIN:
        return {
            "ok": False,
            "reason": f"Data tidak cukup: {n_candles} candle (minimal {MIN_CANDLE_TRAIN})",
            "symbol": symbol,
        }

    # -- Build feature matrix --
    logger.info("[trainer] Building feature matrix (%d candles)...", n_candles)
    feat_df = _build_feature_matrix(raw_df)

    if len(feat_df) < 50:
        return {
            "ok": False,
            "reason": f"Feature matrix terlalu kecil: {len(feat_df)} baris valid",
            "symbol": symbol,
        }

    # -- Prepare X, y --
    X = feat_df[FEATURES].astype(float).values
    y_raw = feat_df["label"].values          # -1, 0, 1
    y = y_raw + 1                            # shift ke 0, 1, 2 (XGBoost multi-class)

    # Time-series split: 70% train, 30% test
    split_idx = int(len(X) * 0.70)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # -- Train --
    logger.info("[trainer] Training XGBoost (%d train / %d test)...", len(X_train), len(X_test))
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # -- Feature importance → bobot --
    raw_imp = model.feature_importances_
    importances = {FEATURES[i]: float(raw_imp[i]) for i in range(len(FEATURES))}

    mean_imp = float(np.mean(raw_imp))
    if mean_imp > 0:
        weights_after = {f: round(float(importances[f]) / mean_imp, 6) for f in FEATURES}
    else:
        weights_after = dict(DEFAULT_WEIGHTS)

    weights_before = load_weights(symbol)
    save_weights(symbol, weights_after)
    logger.info("[trainer] Weights saved for %s", symbol)

    return {
        "ok":              True,
        "symbol":          symbol,
        "interval":        interval,
        "n_candles":       n_candles,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "importances":     importances,
        "weights_before":  weights_before,
        "weights_after":   weights_after,
        "feature_df":      feat_df,
        "raw_df":          raw_df,
        "model":           model,
    }
