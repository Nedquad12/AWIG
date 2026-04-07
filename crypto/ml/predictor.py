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
from ml.weight_manager  import apply_weights, load_weights
try:
    from ml.weight_manager import FEATURES_ALL
except ImportError:
    FEATURES_ALL = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc", "funding", "lsr"]

FEATURES_CORE = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]
_FEATURES_CORE_FALLBACK = FEATURES_CORE 

logger = logging.getLogger(__name__)

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

def _current_scores(df: pd.DataFrame, fund_df, lsr_df) -> dict[str, float]:
    """
    Return skor semua 9 indikator.
    7 core dari kline, funding & LSR dari data real-time terpisah.
    """
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

_SCORE_MODEL_WEIGHT  = 1.0 
_CANDLE_MODEL_WEIGHT = 2.5  
_TOTAL_WEIGHT        = _SCORE_MODEL_WEIGHT + _CANDLE_MODEL_WEIGHT 


def _combine_probas(
    p_long_s:  float, p_short_s:  float, p_neut_s:  float, 
    p_long_c:  float, p_short_c:  float, p_neut_c:  float, 
    regime_w:  float = 1.0,                           
) -> tuple[float, float, float]:

    w_score  = _SCORE_MODEL_WEIGHT * regime_w
    w_candle = _CANDLE_MODEL_WEIGHT

    p_long  = (p_long_s  * w_score + p_long_c  * w_candle) / (w_score + w_candle)
    p_short = (p_short_s * w_score + p_short_c * w_candle) / (w_score + w_candle)
    p_neut  = (p_neut_s  * w_score + p_neut_c  * w_candle) / (w_score + w_candle)

    total = p_long + p_short + p_neut
    if total > 0:
        p_long  /= total
        p_short /= total
        p_neut  /= total

    return round(p_long, 4), round(p_short, 4), round(p_neut, 4)


def predict(train_result: dict) -> dict:
    symbol   = train_result["symbol"]
    interval = train_result["interval"]
    raw_df   = train_result["raw_df"]
    model    = train_result["model"]
    fund_df  = train_result.get("fund_df")
    lsr_df   = train_result.get("lsr_df")

    logger.info("[predictor] Predicting %s %s (dual ML)...", symbol, interval)

    weights = load_weights(symbol)
    scores  = _current_scores(raw_df, fund_df, lsr_df)
    weighted_total = apply_weights(scores, weights)

    feat_df       = train_result["feature_df"]
    available_cols = [c for c in _FEATURES_CORE_FALLBACK if c in feat_df.columns]
    if not available_cols:
        logger.error("[predictor] feat_df columns mismatch — %s", list(feat_df.columns))
        return {"ok": False, "symbol": symbol, "skip": True,
                "skip_reason": "feature_df columns mismatch — re-train required"}

    last_feat_s = feat_df[available_cols].iloc[-1:].astype(float).values
    try:
        proba_s   = model.predict_proba(last_feat_s)[0]
        p_long_s  = float(proba_s[2])
        p_short_s = float(proba_s[0])
        p_neut_s  = float(proba_s[1])
    except Exception as e:
        logger.warning("[predictor] ML1 predict_proba error: %s", e)
        p_long_s = p_short_s = p_neut_s = 1/3
    regime    = "NEUTRAL"
    regime_w  = 1.0
    try:
        from ml.candle_features import detect_regime, get_regime_weight
        regime   = detect_regime(raw_df)
        regime_w = get_regime_weight(regime)
        logger.info("[predictor] %s regime: %s (weight=%.2f)", symbol, regime, regime_w)
    except Exception as e:
        logger.warning("[predictor] Regime detection error: %s", e)

    p_long_c = p_short_c = p_neut_c = 1/3  
    candle_result = train_result.get("candle_result")
    has_candle_model = candle_result is not None and candle_result.get("ok")

    if has_candle_model:
        try:
            from ml.candle_features import get_current_features, CANDLE_FEATURE_NAMES
            candle_model = candle_result["model"]
            feat_c = get_current_features(raw_df)

            if feat_c is not None:
                X_c = np.array([[feat_c[f] for f in CANDLE_FEATURE_NAMES]])
                proba_c   = candle_model.predict_proba(X_c)[0]
                p_long_c  = float(proba_c[2])
                p_short_c = float(proba_c[0])
                p_neut_c  = float(proba_c[1])
                logger.info(
                    "[predictor] ML2 candle proba — long=%.3f short=%.3f neut=%.3f",
                    p_long_c, p_short_c, p_neut_c,
                )
            else:
                logger.warning("[predictor] get_current_features return None untuk %s", symbol)
                has_candle_model = False
        except Exception as e:
            logger.warning("[predictor] ML2 predict error: %s", e)
            has_candle_model = False
    if has_candle_model:
        p_long, p_short, p_neut = _combine_probas(
            p_long_s, p_short_s, p_neut_s,
            p_long_c, p_short_c, p_neut_c,
            regime_w=regime_w,
        )
        model_used = "dual_ml"
    else:
        p_long, p_short, p_neut = p_long_s, p_short_s, p_neut_s
        model_used = "scoring_only"
        logger.info("[predictor] %s fallback ke ML 1 saja", symbol)

    logger.info(
        "[predictor] %s combined — long=%.3f short=%.3f neut=%.3f [%s, regime=%s]",
        symbol, p_long, p_short, p_neut, model_used, regime,
    )

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

    skip        = confidence < CONFIDENCE_MIN or direction == "NEUTRAL"
    skip_reason = ""

    return {
        "ok":              True,
        "symbol":          symbol,
        "interval":        interval,
        "direction":       direction,
        "confidence":      round(confidence, 4),
        "p_long":          round(p_long,   4),
        "p_short":         round(p_short,  4),
        "p_neutral":       round(p_neut,   4),
        "p_long_scoring":  round(p_long_s,  4),
        "p_short_scoring": round(p_short_s, 4),
        "p_long_candle":   round(p_long_c,  4),
        "p_short_candle":  round(p_short_c, 4),
        "regime":          regime,
        "regime_weight":   round(regime_w, 3),
        "model_used":      model_used,
        "predicted_price": predicted_price,
        "current_price":   current_price,
        "weighted_total":  round(weighted_total, 4),
        "scores":          scores,
        "weights":         weights,
        "context_df":      context_df,
        "skip":            skip,
        "skip_reason":     skip_reason,
    }
