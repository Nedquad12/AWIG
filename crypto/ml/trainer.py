# =============================================================
# ml/trainer.py — XGBoost trainer untuk adjust bobot indikator
# =============================================================

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CANDLE_LIMIT, DEFAULT_INTERVAL,
    LABEL_UP_PCT, LABEL_DOWN_PCT,
    LOOKAHEAD, MIN_CANDLE_TRAIN,
    DEFAULT_INTERVAL,
)
from indicators.binance_fetcher import get_df
from indicators import (
    score_vsa, score_fsa, score_vfa,
    score_rsi, score_macd, score_ma, score_wcc,
)
from indicators.funding import fetch_funding_rate, score_funding
from indicators.lsr     import fetch_lsr, score_lsr
from ml.weight_manager  import DEFAULT_WEIGHTS, FEATURES, load_weights, save_weights

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Hitung skor semua indikator per-candle (rolling window)
# ------------------------------------------------------------------

def _score_at(df: pd.DataFrame, i: int, fund_df=None, lsr_df=None) -> dict[str, float]:
    """
    Hitung skor semua indikator menggunakan data df[:i+1].
    Minimal window = 210 candle (kebutuhan MA200).
    funding dan lsr dipakai nilai tetap (data terbaru) karena
    tidak tersedia per-candle historis yang efisien.
    """
    window = df.iloc[:i + 1]
    if len(window) < 210:
        return {f: 0.0 for f in FEATURES}

    scores = {
        "vsa":  float(score_vsa(window)),
        "fsa":  float(score_fsa(window)),
        "vfa":  float(score_vfa(window)),
        "rsi":  float(score_rsi(window)),
        "macd": float(score_macd(window)),
        "ma":   float(score_ma(window)),
        "wcc":  float(score_wcc(window)),
    }

    # funding & lsr: pakai nilai terbaru (konstan per sesi training)
    scores["funding"] = float(score_funding(fund_df)) if fund_df is not None and not fund_df.empty else 0.0
    scores["lsr"]     = float(score_lsr(lsr_df))     if lsr_df  is not None and not lsr_df.empty  else 0.0

    return scores


def _build_feature_matrix(df: pd.DataFrame, fund_df, lsr_df) -> pd.DataFrame:
    """Bangun feature matrix: tiap baris = skor indikator pada candle ke-i."""
    rows   = []
    prices = df["close"].values

    for i in range(len(df)):
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

        row = _score_at(df, i, fund_df, lsr_df)
        row["label"] = label
        row["price"] = float(prices[i])
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result[result["label"].notna()].copy()
    result = result[result[FEATURES].any(axis=1)].copy()
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
    Semua 9 indikator (vsa/fsa/vfa/rsi/macd/ma/wcc/funding/lsr) ikut training.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"ok": False, "reason": "XGBoost belum terinstall. Jalankan: pip install xgboost"}

    symbol = symbol.upper()

    # -- Fetch kline --
    logger.info("[trainer] Fetch %s candle %s %s", limit, symbol, interval)
    raw_df = get_df(symbol, interval=interval, limit=limit)
    n_candles = len(raw_df)

    if n_candles < MIN_CANDLE_TRAIN:
        return {
            "ok": False,
            "reason": f"Data tidak cukup: {n_candles} candle (minimal {MIN_CANDLE_TRAIN})",
            "symbol": symbol,
        }

    # -- Fetch funding & lsr (sekali saja untuk efisiensi) --
    logger.info("[trainer] Fetch funding rate & LSR untuk %s...", symbol)
    try:
        fund_df = fetch_funding_rate(symbol, limit=90)
    except Exception as e:
        logger.warning("[trainer] Funding rate fetch gagal untuk %s: %s", symbol, e)
        fund_df = None

    try:
        lsr_df = fetch_lsr(symbol, interval=interval, limit=96)
    except Exception as e:
        logger.warning("[trainer] LSR fetch gagal untuk %s: %s", symbol, e)
        lsr_df = None

    # -- Build feature matrix --
    logger.info("[trainer] Building feature matrix (%d candles)...", n_candles)
    feat_df = _build_feature_matrix(raw_df, fund_df, lsr_df)

    if len(feat_df) < 50:
        return {
            "ok": False,
            "reason": f"Feature matrix terlalu kecil: {len(feat_df)} baris valid",
            "symbol": symbol,
        }

    # -- Prepare X, y --
    X    = feat_df[FEATURES].astype(float).values
    y    = feat_df["label"].values + 1   # shift -1,0,1 → 0,1,2

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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # -- Feature importance → bobot --
    raw_imp    = model.feature_importances_
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
        "ok":             True,
        "symbol":         symbol,
        "interval":       interval,
        "n_candles":      n_candles,
        "n_train":        len(X_train),
        "n_test":         len(X_test),
        "importances":    importances,
        "weights_before": weights_before,
        "weights_after":  weights_after,
        "feature_df":     feat_df,
        "raw_df":         raw_df,
        "fund_df":        fund_df,
        "lsr_df":         lsr_df,
        "model":          model,
    }
