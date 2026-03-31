# =============================================================
# ml/predictor.py — Prediksi + persiapan konteks AI
#
# FIXES:
#   1. Model hanya pakai FEATURES_CORE (7) — sesuai dengan training
#   2. Funding & LSR dipakai sebagai POST-PREDICTION FILTER,
#      bukan sebagai fitur model. Logika:
#        - Model output arah (LONG/SHORT/NEUTRAL)
#        - Funding & LSR bisa override ke SKIP jika sangat berlawanan
#   3. Weighted total untuk display/scanner masih pakai semua 9 indikator
#      (ini bukan untuk model — ini untuk human-readable summary)
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
from ml.weight_manager  import apply_weights, load_weights
try:
    from ml.weight_manager import FEATURES_ALL
except ImportError:
    FEATURES_ALL = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc", "funding", "lsr"]

# Hardcoded — tidak import dari weight_manager untuk hindari version mismatch
FEATURES_CORE = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]
_FEATURES_CORE_FALLBACK = FEATURES_CORE  # alias untuk backward compat

logger = logging.getLogger(__name__)

# Threshold override: jika funding/LSR sangat berlawanan dengan arah model
# → skip trade (terlalu berisiko melawan sentiment ekstrem)
OVERRIDE_THRESHOLD = 2   # abs(funding_score) atau abs(lsr_score) >= ini → pertimbangkan override


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
# Hitung skor semua indikator pada candle terakhir
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Cek apakah funding/LSR sangat berlawanan dengan arah prediksi
# ------------------------------------------------------------------

def _check_sentiment_override(direction: str, scores: dict[str, float]) -> tuple[bool, str]:
    """
    Return (should_override, reason).
    Override ke SKIP jika sentiment ekstrem berlawanan dengan arah model.

    Logic:
      - LONG prediction tapi funding sangat negatif (< -OVERRIDE_THRESHOLD) → pasar terlalu short?
        Tidak, ini sebenarnya bullish untuk LONG (short squeeze potential) → jangan override
      - LONG prediction tapi funding sangat positif (> +OVERRIDE_THRESHOLD) → pasar terlalu long
        Funding positif tinggi = contrarian bearish → override jika LONG
      - SHORT prediction tapi funding sangat negatif → pasar terlalu short = contrarian bullish → override jika SHORT

    Simplified rule:
      Jika sign(funding_score) SAMA dengan sign(direction) → tidak override
      (funding_score positif = bearish signal, jadi berlawanan dengan LONG)

    Ingat: score_funding positif = BULLISH (funding negatif = short dominan = potensi squeeze)
    """
    fund_score = scores.get("funding", 0.0)
    lsr_score  = scores.get("lsr", 0.0)

    if direction == "LONG":
        # Untuk LONG, kita butuh fund & lsr tidak terlalu negatif (bearish extrem)
        if fund_score <= -OVERRIDE_THRESHOLD and lsr_score <= -OVERRIDE_THRESHOLD:
            return True, (
                f"Funding score {fund_score:+.0f} dan LSR score {lsr_score:+.0f} "
                f"sangat bearish — berlawanan dengan prediksi LONG"
            )
    elif direction == "SHORT":
        # Untuk SHORT, kita butuh fund & lsr tidak terlalu positif (bullish extrem)
        if fund_score >= OVERRIDE_THRESHOLD and lsr_score >= OVERRIDE_THRESHOLD:
            return True, (
                f"Funding score {fund_score:+.0f} dan LSR score {lsr_score:+.0f} "
                f"sangat bullish — berlawanan dengan prediksi SHORT"
            )

    return False, ""


# ------------------------------------------------------------------
# Estimasi harga ke depan
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

# Bobot kombinasi dua model
_SCORE_MODEL_WEIGHT  = 1.0   # ML 1: scoring model
_CANDLE_MODEL_WEIGHT = 2.5   # ML 2: candle model (lebih besar)
_TOTAL_WEIGHT        = _SCORE_MODEL_WEIGHT + _CANDLE_MODEL_WEIGHT   # 3.5


def _combine_probas(
    p_long_s:  float, p_short_s:  float, p_neut_s:  float,  # scoring model
    p_long_c:  float, p_short_c:  float, p_neut_c:  float,  # candle model
    regime_w:  float = 1.0,                                   # regime weight untuk scoring
) -> tuple[float, float, float]:
    """
    Gabungkan probabilitas dua model dengan bobot tertimbang.

    Scoring model weight dikali regime_w (dikurangi kalau SIDEWAYS).
    Candle model weight tetap — ADX sudah masuk sebagai fitur di dalamnya.

    Returns: (p_long, p_short, p_neutral) yang sudah dinormalisasi
    """
    w_score  = _SCORE_MODEL_WEIGHT * regime_w
    w_candle = _CANDLE_MODEL_WEIGHT

    p_long  = (p_long_s  * w_score + p_long_c  * w_candle) / (w_score + w_candle)
    p_short = (p_short_s * w_score + p_short_c * w_candle) / (w_score + w_candle)
    p_neut  = (p_neut_s  * w_score + p_neut_c  * w_candle) / (w_score + w_candle)

    # Normalisasi agar total = 1.0
    total = p_long + p_short + p_neut
    if total > 0:
        p_long  /= total
        p_short /= total
        p_neut  /= total

    return round(p_long, 4), round(p_short, 4), round(p_neut, 4)


def predict(train_result: dict) -> dict:
    """
    Hitung prediksi dari dua model:
      ML 1 (Scoring Model) : 7 indikator rule-based  → bobot 1x × regime_weight
      ML 2 (Candle Model)  : 14 raw price features   → bobot 2.5x

    Final confidence = weighted average dua model.
    Funding & LSR sebagai post-prediction sentiment filter.
    """
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

    # ── ML 1: Scoring Model ───────────────────────────────────────
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

    # ── Regime detection (untuk adjust bobot scoring model) ───────
    regime    = "NEUTRAL"
    regime_w  = 1.0
    try:
        from ml.candle_features import detect_regime, get_regime_weight
        regime   = detect_regime(raw_df)
        regime_w = get_regime_weight(regime)
        logger.info("[predictor] %s regime: %s (weight=%.2f)", symbol, regime, regime_w)
    except Exception as e:
        logger.warning("[predictor] Regime detection error: %s", e)

    # ── ML 2: Candle Model ────────────────────────────────────────
    p_long_c = p_short_c = p_neut_c = 1/3   # default jika candle model tidak ada
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

    # ── Combine ───────────────────────────────────────────────────
    if has_candle_model:
        p_long, p_short, p_neut = _combine_probas(
            p_long_s, p_short_s, p_neut_s,
            p_long_c, p_short_c, p_neut_c,
            regime_w=regime_w,
        )
        model_used = "dual_ml"
    else:
        # Fallback ke ML 1 saja
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

    if not skip and direction != "NEUTRAL":
        override, override_reason = _check_sentiment_override(direction, scores)
        if override:
            skip        = True
            skip_reason = f"Sentiment override: {override_reason}"
            logger.info("[predictor] %s skipped — %s", symbol, skip_reason)

    return {
        "ok":              True,
        "symbol":          symbol,
        "interval":        interval,
        "direction":       direction,
        "confidence":      round(confidence, 4),
        "p_long":          round(p_long,   4),
        "p_short":         round(p_short,  4),
        "p_neutral":       round(p_neut,   4),
        # Detail per model (untuk debugging & AI context)
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
