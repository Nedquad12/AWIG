import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    CANDLE_LIMIT, DEFAULT_INTERVAL,
    LOOKAHEAD, MIN_CANDLE_TRAIN,
)
from indicators.binance_fetcher import get_df
from indicators import (
    score_vsa, score_fsa, score_vfa,
    score_rsi, score_macd, score_ma, score_wcc,
)
from indicators.funding import fetch_funding_rate
from indicators.lsr     import fetch_lsr
from ml.weight_manager  import DEFAULT_WEIGHTS, load_weights, save_weights
from ml.regime_detector import detect_regime, REGIME_PARAMS
from ml.wfv             import TRAIN_END_IDX

FEATURES_CORE    = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]
ATR_PERIOD_LABEL = 14
MAX_CLASS_RATIO  = 4.0

logger = logging.getLogger(__name__)

def _rolling_atr(df: pd.DataFrame, period: int = ATR_PERIOD_LABEL) -> np.ndarray:
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    tr = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i]  - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    atr = np.full(len(df), np.nan)
    for i in range(period, len(df)):
        atr[i] = float(np.mean(tr[i - period + 1: i + 1]))

    return atr

def _score_at(df: pd.DataFrame, i: int) -> dict[str, float]:
    window = df.iloc[: i + 1]
    if len(window) < 210:
        return {f: 0.0 for f in FEATURES_CORE}
    return {
        "vsa":  float(score_vsa(window)),
        "fsa":  float(score_fsa(window)),
        "vfa":  float(score_vfa(window)),
        "rsi":  float(score_rsi(window)),
        "macd": float(score_macd(window)),
        "ma":   float(score_ma(window)),
        "wcc":  float(score_wcc(window)),
    }

def _build_feature_matrix(
    df: pd.DataFrame,
    atr_label_mult: float,
    end_idx: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    prices         = df["close"].values
    atr_arr        = _rolling_atr(df)
    rows           = []
    labels_aligned = np.full(len(df), np.nan)

    max_i = (end_idx - 1) if end_idx is not None else (len(df) - 1)

    for i in range(len(df)):
        if i > max_i:
            break
        if np.isnan(atr_arr[i]):
            continue
        if i + LOOKAHEAD >= len(prices):
            continue
        atr_i = atr_arr[i]
        if atr_i <= 0:
            continue

        ret       = (prices[i + LOOKAHEAD] - prices[i]) / prices[i]
        threshold = (atr_label_mult * atr_i) / prices[i]

        label = 1 if ret >= threshold else (-1 if ret <= -threshold else 0)

        labels_aligned[i] = label
        row = _score_at(df, i)
        row["label"] = label
        row["price"] = float(prices[i])
        rows.append(row)

    if not rows:
        logger.warning(
            "[trainer] 0 rows terbentuk (end_idx=%s, total=%d, atr_mult=%.1f)",
            end_idx, len(df), atr_label_mult,
        )
        return pd.DataFrame(columns=FEATURES_CORE + ["label", "price"]), labels_aligned

    result  = pd.DataFrame(rows)
    missing = [c for c in FEATURES_CORE if c not in result.columns]
    if missing:
        logger.error("[trainer] feat_df missing columns: %s", missing)
        return pd.DataFrame(columns=FEATURES_CORE + ["label", "price"]), labels_aligned

    result = result[result[FEATURES_CORE].any(axis=1)].copy()
    result["label"] = result["label"].astype(int)
    return result.reset_index(drop=True), labels_aligned

def _balance_classes(train_df: pd.DataFrame) -> pd.DataFrame:
    counts      = train_df["label"].value_counts()
    min_size    = int(counts.min())
    max_allowed = int(min_size * MAX_CLASS_RATIO)
    parts = []
    for label_val, count in counts.items():
        subset = train_df[train_df["label"] == label_val]
        if count > max_allowed:
            subset = subset.sample(n=max_allowed, random_state=42)
            logger.info("[trainer] Undersample class %d: %d → %d", label_val, count, max_allowed)
        parts.append(subset)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def _check_classes(feat_df: pd.DataFrame, symbol: str) -> bool:
    missing = {-1, 0, 1} - set(feat_df["label"].unique())
    if missing:
        logger.warning("[trainer] %s — class hilang: %s. Skip.", symbol, sorted(missing))
        return False
    return True

def train(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    limit: int = CANDLE_LIMIT,
) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"ok": False, "reason": "XGBoost belum terinstall. pip install xgboost"}

    symbol = symbol.upper()

    logger.info("[trainer] Fetch %d candle %s %s...", limit, symbol, interval)
    raw_df    = get_df(symbol, interval=interval, limit=limit)
    n_candles = len(raw_df)

    if n_candles < MIN_CANDLE_TRAIN:
        return {
            "ok":     False,
            "reason": f"Data tidak cukup: {n_candles} candle (minimal {MIN_CANDLE_TRAIN})",
            "symbol": symbol,
        }

    fund_df = lsr_df = None
    try:
        fund_df = fetch_funding_rate(symbol, limit=90)
    except Exception as e:
        logger.warning("[trainer] Funding gagal %s: %s", symbol, e)
    try:
        lsr_df = fetch_lsr(symbol, interval=interval, limit=96)
    except Exception as e:
        logger.warning("[trainer] LSR gagal %s: %s", symbol, e)

    effective_train_end = min(TRAIN_END_IDX, n_candles)
    train_raw_window    = raw_df.iloc[:effective_train_end]
    regime_info         = detect_regime(train_raw_window)
    regime              = regime_info["regime"]
    regime_params       = regime_info["params"]
    atr_label_mult      = regime_params["atr_label_mult"]
    kelly_multiplier    = regime_params["kelly_multiplier"]

    logger.info(
        "[trainer] %s — regime=%s | ATR_MULT=%.1f | kelly=%.2f",
        symbol, regime, atr_label_mult, kelly_multiplier,
    )

    logger.info("[trainer] Building train feat_df (0–%d, ATR_MULT=%.1f)...",
                effective_train_end, atr_label_mult)
    train_feat_df, labels_aligned = _build_feature_matrix(
        raw_df, atr_label_mult, end_idx=effective_train_end
    )

    if len(train_feat_df) < 50:
        return {
            "ok":     False,
            "reason": f"Feature matrix terlalu kecil: {len(train_feat_df)} rows",
            "symbol": symbol,
        }

    label_counts = train_feat_df["label"].value_counts().to_dict()
    logger.info("[trainer] %s — %d rows | labels: %s", symbol, len(train_feat_df), label_counts)

    if not _check_classes(train_feat_df, symbol):
        return {
            "ok":     False,
            "reason": f"Class tidak lengkap di {symbol}: {label_counts}",
            "symbol": symbol,
        }

    train_bal       = _balance_classes(train_feat_df)
    balanced_counts = train_bal["label"].value_counts().to_dict()

    X_train = train_bal[FEATURES_CORE].astype(float).values
    y_train = train_bal["label"].values + 1   

    class_counts  = np.bincount(y_train, minlength=3)
    total         = len(y_train)
    class_w       = {c: total / (3.0 * max(class_counts[c], 1)) for c in range(3)}
    sample_weight = np.array([class_w[y] for y in y_train])

    logger.info("[trainer] Training XGBoost (%d rows, %s)...", len(X_train), symbol)
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
    model.fit(X_train, y_train, sample_weight=sample_weight)

    raw_imp     = model.feature_importances_
    importances = {FEATURES_CORE[i]: float(raw_imp[i]) for i in range(len(FEATURES_CORE))}
    mean_imp    = float(np.mean(raw_imp))

    weights_after = (
        {f: round(float(importances[f]) / mean_imp, 6) for f in FEATURES_CORE}
        if mean_imp > 0 else
        {f: DEFAULT_WEIGHTS[f] for f in FEATURES_CORE}
    )
    weights_before = load_weights(symbol)
    save_weights(symbol, weights_after)
    logger.info("[trainer] Weights saved for %s (regime=%s)", symbol, regime)

    logger.info("[trainer] Building full feat_df (semua candle)...")
    full_feat_df, _ = _build_feature_matrix(raw_df, atr_label_mult, end_idx=None)

    candle_result = None
    try:
        from ml.candle_model import train_candle_model, backtest_candle_model
        logger.info("[trainer] Training candle model %s (0–%d)...", symbol, effective_train_end)
        candle_result = train_candle_model(
            symbol,
            raw_df.iloc[:effective_train_end].reset_index(drop=True),
            labels_aligned[:effective_train_end],
        )
        if candle_result["ok"]:
            candle_bt = backtest_candle_model(candle_result)
            candle_result["backtest"] = candle_bt
            logger.info(
                "[trainer] Candle %s — acc=%.1f%% wr_up=%.1f%% wr_dn=%.1f%%",
                symbol,
                candle_bt["accuracy"]   * 100,
                candle_bt["winrate_up"] * 100,
                candle_bt["winrate_dn"] * 100,
            )
    except Exception as e:
        logger.warning("[trainer] Candle error %s: %s", symbol, e)

    return {
        "ok":               True,
        "symbol":           symbol,
        "interval":         interval,
        "n_candles":        n_candles,
        "n_train":          len(X_train),
        "train_end_idx":    effective_train_end,
        "regime":           regime,
        "regime_info":      regime_info,
        "atr_label_mult":   atr_label_mult,
        "kelly_multiplier": kelly_multiplier,
        "label_counts":     label_counts,
        "balanced_counts":  balanced_counts,
        "had_missing_class":False,
        "importances":      importances,
        "weights_before":   weights_before,
        "weights_after":    weights_after,
        "feature_df":       full_feat_df,
        "raw_df":           raw_df,
        "fund_df":          fund_df,
        "lsr_df":           lsr_df,
        "model":            model,
        "candle_result":    candle_result,
        "labels_aligned":   labels_aligned,
    }
