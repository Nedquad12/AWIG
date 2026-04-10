"""
ml/candle_model.py — Trainer untuk ML 2 (Candle Model).

ML 2 belajar dari raw price features (14 fitur dari 20 candle terakhir),
bukan dari skor indikator. Ini complementary dengan ML 1 (Scoring Model):
  - ML 1 tahu "indikator apa yang aktif"
  - ML 2 tahu "price action seperti apa yang terjadi"

Label: sama persis dengan ML 1 — ATR-based, dari trainer.py.
Disupply via train_result["labels_array"] agar tidak compute ulang.

Output:
  candle model (XGBClassifier)
  feature importances
  backtest metrics
"""

import logging
import numpy as np
import pandas as pd

from ml.candle_features import (
    CANDLE_FEATURE_NAMES,
    build_feature_matrix,
    detect_regime,
)

logger = logging.getLogger(__name__)

# Sama dengan trainer.py
MAX_CLASS_RATIO = 4.0


# ------------------------------------------------------------------
# Balance classes via undersample (reuse logic dari trainer.py)
# ------------------------------------------------------------------

def _balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    counts      = df["label"].value_counts()
    min_size    = int(counts.min())
    max_allowed = int(min_size * MAX_CLASS_RATIO)
    parts = []
    for label_val, count in counts.items():
        subset = df[df["label"] == label_val]
        if count > max_allowed:
            subset = subset.sample(n=max_allowed, random_state=42)
        parts.append(subset)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def _check_classes(df: pd.DataFrame, symbol: str) -> bool:
    present = set(df["label"].unique())
    missing = {-1, 0, 1} - present
    if missing:
        logger.warning(
            "[candle_model] %s — class hilang: %s. Skip candle model.",
            symbol, sorted(missing),
        )
        return False
    return True


# ------------------------------------------------------------------
# Public: train_candle_model
# ------------------------------------------------------------------

def train_candle_model(
    symbol: str,
    raw_df: pd.DataFrame,
    labels_aligned: np.ndarray,
) -> dict:
    """
    Train XGBoost candle model.

    Args:
        symbol         : nama koin
        raw_df         : OHLCV DataFrame (1000 candle, ascending)
        labels_aligned : array label (-1/0/1) per candle, sama panjang raw_df.
                         Diambil dari trainer.py (ATR-based labels).
                         NaN untuk candle yang tidak punya label valid.

    Returns:
        {
          "ok": True/False,
          "model": XGBClassifier,
          "importances": dict,
          "feat_df": DataFrame,
          "regime": str,
          "n_train": int, "n_test": int,
        }
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"ok": False, "reason": "XGBoost tidak terinstall"}

    symbol = symbol.upper()

    # Build feature matrix dari raw OHLCV + labels dari ML 1
    logger.info("[candle_model] Building candle feature matrix untuk %s...", symbol)
    feat_df = build_feature_matrix(raw_df, labels_aligned)

    if len(feat_df) < 50:
        return {
            "ok":     False,
            "reason": f"Candle feature matrix terlalu kecil: {len(feat_df)} baris",
            "symbol": symbol,
        }

    label_counts = feat_df["label"].value_counts().to_dict()
    logger.info("[candle_model] Label distribution %s: %s", symbol, label_counts)

    # Split train/test
    split_idx = int(len(feat_df) * 0.70)
    train_df  = feat_df.iloc[:split_idx].copy()
    test_df   = feat_df.iloc[split_idx:].copy()

    if not _check_classes(train_df, symbol):
        return {
            "ok":     False,
            "reason": f"Class tidak lengkap di candle train set: {label_counts}",
            "symbol": symbol,
        }

    train_df = _balance_classes(train_df)

    X_train = train_df[CANDLE_FEATURE_NAMES].astype(float).values
    y_train = train_df["label"].values + 1   # shift -1,0,1 → 0,1,2
    X_test  = test_df[CANDLE_FEATURE_NAMES].astype(float).values
    y_test  = test_df["label"].values + 1

    # Sample weight
    class_counts  = np.bincount(y_train, minlength=3)
    total         = len(y_train)
    class_w       = {c: total / (3.0 * max(class_counts[c], 1)) for c in range(3)}
    sample_weight = np.array([class_w[y] for y in y_train])

    logger.info(
        "[candle_model] Training XGBoost candle model (%d train / %d test) untuk %s...",
        len(X_train), len(X_test), symbol,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,            # sedikit lebih dalam dari scoring model
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
        sample_weight=sample_weight,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    raw_imp     = model.feature_importances_
    importances = {
        CANDLE_FEATURE_NAMES[i]: float(raw_imp[i])
        for i in range(len(CANDLE_FEATURE_NAMES))
    }

    # Detect current regime
    regime = detect_regime(raw_df)
    logger.info("[candle_model] %s regime: %s", symbol, regime)

    return {
        "ok":          True,
        "symbol":      symbol,
        "model":       model,
        "importances": importances,
        "feat_df":     feat_df,
        "regime":      regime,
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "label_counts": label_counts,
    }


# ------------------------------------------------------------------
# Backtest candle model
# ------------------------------------------------------------------

def backtest_candle_model(candle_train_result: dict) -> dict:
    """
    Evaluasi sederhana candle model di test set.
    Return winrate long/short dan accuracy.
    """
    feat_df = candle_train_result["feat_df"]
    model   = candle_train_result["model"]
    symbol  = candle_train_result["symbol"]

    split_idx = int(len(feat_df) * 0.70)
    test_df   = feat_df.iloc[split_idx:].copy()

    if len(test_df) == 0:
        return {"accuracy": 0.0, "winrate_up": 0.0, "winrate_dn": 0.0}

    X_test = test_df[CANDLE_FEATURE_NAMES].astype(float).values
    y_test = test_df["label"].values   # -1, 0, 1

    proba  = model.predict_proba(X_test)   # (n, 3) → [P(down), P(neutral), P(up)]

    # Signal: prediksi up jika P(up) > 0.5, down jika P(down) > 0.5
    p_up   = proba[:, 2]
    p_down = proba[:, 0]

    pred_up   = p_up   >= 0.5
    pred_down = p_down >= 0.5

    n_up = int(pred_up.sum())
    n_dn = int(pred_down.sum())

    tp_up = int(((pred_up)   & (y_test == 1)).sum())
    tp_dn = int(((pred_down) & (y_test == -1)).sum())

    wr_up  = tp_up / n_up if n_up > 0 else 0.0
    wr_dn  = tp_dn / n_dn if n_dn > 0 else 0.0
    n_sig  = n_up + n_dn
    acc    = (tp_up + tp_dn) / n_sig if n_sig > 0 else 0.0

    logger.info(
        "[candle_model] %s backtest — acc=%.1f%% wr_up=%.1f%% wr_dn=%.1f%%",
        symbol, acc * 100, wr_up * 100, wr_dn * 100,
    )

    return {
        "accuracy":   round(acc,   4),
        "winrate_up": round(wr_up, 4),
        "winrate_dn": round(wr_dn, 4),
        "n_signal_up": n_up,
        "n_signal_dn": n_dn,
    }
