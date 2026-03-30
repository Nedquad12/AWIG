# =============================================================
# ml/predictor.py — Prediksi + persiapan konteks AI
# =============================================================

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIDENCE_MIN, DEFAULT_INTERVAL, LOOKAHEAD
from indicators import (
    score_vsa, score_fsa, score_vfa,
    score_rsi, score_macd, score_ma, score_wcc,
)
from indicators.funding import score_funding
from indicators.lsr     import score_lsr
from ml.weight_manager  import FEATURES, apply_weights, load_weights

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Enrich DataFrame dengan kolom teknikal untuk AI
# ------------------------------------------------------------------

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["close"]
    vol   = df["volume"]
    freq  = df["transactions"]

    df["ma10"]  = close.rolling(10).mean()
    df["ma20"]  = close.rolling(20).mean()
    df["ma50"]  = close.rolling(50).mean()

    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    df["vol_ma10"]  = vol.rolling(10).mean()
    df["vol_ma20"]  = vol.rolling(20).mean()
    df["freq_ma10"] = freq.rolling(10).mean()
    df["freq_ma20"] = freq.rolling(20).mean()

    return df


# ------------------------------------------------------------------
# Hitung skor semua 9 indikator pada candle terakhir
# ------------------------------------------------------------------

def _current_scores(df: pd.DataFrame, fund_df, lsr_df) -> dict[str, float]:
    scores = {
        "vsa":  float(score_vsa(df)),
        "fsa":  float(score_fsa(df)),
        "vfa":  float(score_vfa(df)),
        "rsi":  float(score_rsi(df)),
        "macd": float(score_macd(df)),
        "ma":   float(score_ma(df)),
        "wcc":  float(score_wcc(df)),
    }
    scores["funding"] = float(score_funding(fund_df)) if fund_df is not None and not fund_df.empty else 0.0
    scores["lsr"]     = float(score_lsr(lsr_df))     if lsr_df  is not None and not lsr_df.empty  else 0.0
    return scores


# ------------------------------------------------------------------
# Estimasi harga 3 candle ke depan
# ------------------------------------------------------------------

def _estimate_price(raw_df: pd.DataFrame, direction: str) -> float:
    closes = raw_df["close"].values
    if len(closes) < 50:
        return float(closes[-1])

    returns = []
    for i in range(len(closes) - LOOKAHEAD - 1):
        ret = (closes[i + LOOKAHEAD] - closes[i]) / closes[i]
        if direction == "LONG"  and ret > 0:
            returns.append(ret)
        elif direction == "SHORT" and ret < 0:
            returns.append(ret)

    if not returns:
        return float(closes[-1])

    avg_ret = float(np.median(returns))
    return round(float(closes[-1]) * (1 + avg_ret), 6)


# ------------------------------------------------------------------
# Public: predict
# ------------------------------------------------------------------

def predict(train_result: dict) -> dict:
    """
    Hitung prediksi berdasarkan hasil training.
    Menggunakan semua 9 indikator termasuk funding & lsr.
    """
    symbol   = train_result["symbol"]
    interval = train_result["interval"]
    raw_df   = train_result["raw_df"]
    model    = train_result["model"]
    fund_df  = train_result.get("fund_df")
    lsr_df   = train_result.get("lsr_df")

    logger.info("[predictor] Predicting %s %s...", symbol, interval)

    weights        = load_weights(symbol)
    scores         = _current_scores(raw_df, fund_df, lsr_df)
    weighted_total = apply_weights(scores, weights)

    # Confidence via model.predict_proba
    feat_df   = train_result["feature_df"]
    last_feat = feat_df[FEATURES].iloc[-1:].astype(float).values

    try:
        proba   = model.predict_proba(last_feat)[0]  # [P(turun), P(netral), P(naik)]
        p_long  = float(proba[2])
        p_short = float(proba[0])
        p_neut  = float(proba[1])
    except Exception as e:
        logger.warning("[predictor] predict_proba error: %s", e)
        p_long = p_short = p_neut = 1/3

    if p_long >= p_short and p_long >= p_neut:
        direction  = "LONG"
        confidence = p_long
    elif p_short >= p_long and p_short >= p_neut:
        direction  = "SHORT"
        confidence = p_short
    else:
        direction  = "NEUTRAL"
        confidence = p_neut

    current_price   = float(raw_df["close"].iloc[-1])
    predicted_price = _estimate_price(raw_df, direction)
    context_df      = enrich_df(raw_df)
    skip            = confidence < CONFIDENCE_MIN or direction == "NEUTRAL"

    return {
        "ok":              True,
        "symbol":          symbol,
        "interval":        interval,
        "direction":       direction,
        "confidence":      round(confidence, 4),
        "p_long":          round(p_long,  4),
        "p_short":         round(p_short, 4),
        "p_neutral":       round(p_neut,  4),
        "predicted_price": predicted_price,
        "current_price":   current_price,
        "weighted_total":  round(weighted_total, 4),
        "scores":          scores,
        "weights":         weights,
        "context_df":      context_df,
        "skip":            skip,
    }
