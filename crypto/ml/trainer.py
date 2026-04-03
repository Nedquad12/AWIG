# =============================================================
# ml/trainer.py — XGBoost trainer untuk adjust bobot indikator
#
# FIXES:
#   1. Data leakage: funding & LSR tidak lagi dipakai sebagai fitur
#      rolling per-candle (karena kita tidak punya history per-candle
#      untuk keduanya). Keduanya dipindahkan ke filter post-prediction.
#   2. ATR-based labeling: threshold label tidak lagi flat 0.5%,
#      melainkan relatif terhadap ATR tiap candle → label lebih valid.
#   3. Synthetic data injection dihapus → diganti class_weight balanced
#      + undersample class dominan. Tidak ada lagi "mean features = class X".
# =============================================================

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

# Hardcoded — tidak import dari weight_manager untuk hindari version mismatch.
# Ini adalah 7 indikator yang bisa dihitung rolling per-candle tanpa leakage.
# funding & lsr TIDAK masuk sini karena tidak ada historical per-candle data.
FEATURES_CORE = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]

logger = logging.getLogger(__name__)

# ATR multiplier untuk threshold label
# Label UP   jika return 3 candle ≥ +ATR_LABEL_MULT × ATR
# Label DOWN jika return 3 candle ≤ -ATR_LABEL_MULT × ATR
ATR_LABEL_MULT   = 0.75   # lebih konservatif dari 1x ATR, tapi jauh lebih valid dari flat 0.5%
ATR_PERIOD_LABEL = 14

# Max rasio class imbalance sebelum undersample
MAX_CLASS_RATIO  = 4.0


# ------------------------------------------------------------------
# ATR per-candle (rolling, tidak ada lookahead bias)
# ------------------------------------------------------------------

def _rolling_atr(df: pd.DataFrame, period: int = ATR_PERIOD_LABEL) -> np.ndarray:
    """
    Hitung ATR untuk setiap baris secara rolling (pakai data t ke belakang saja).
    Return array panjang len(df), NaN untuk baris pertama yg belum cukup data.
    """
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


# ------------------------------------------------------------------
# Hitung skor 7 indikator pada candle ke-i
# PENTING: funding & LSR DIHAPUS dari fitur rolling karena kita tidak
# punya history per-candle untuk keduanya → data leakage.
# Keduanya tetap dipakai sebagai filter di predictor & analyst,
# tapi BUKAN sebagai fitur ML.
# ------------------------------------------------------------------

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


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build feature matrix dengan ATR-based labeling.
    Tidak ada leakage: label dihitung dari close harga masa depan,
    threshold dari ATR masa lalu (sudah diketahui saat candle i).

    Returns:
        (feat_df, labels_aligned)
        labels_aligned: array label per candle (len=len(df)),
                        NaN untuk candle yang di-skip.
                        Dipakai oleh candle_model agar label konsisten.
    """
    prices  = df["close"].values
    atr_arr = _rolling_atr(df)
    rows    = []

    # labels_aligned: satu nilai per candle, NaN jika di-skip
    labels_aligned = np.full(len(df), np.nan)

    for i in range(len(df)):
        # Butuh ATR valid dan cukup lookahead
        if np.isnan(atr_arr[i]):
            continue
        if i + LOOKAHEAD >= len(df):
            continue

        atr_i = atr_arr[i]
        if atr_i <= 0:
            continue

        # Label berbasis ATR — threshold adaptif per volatilitas
        ret = (prices[i + LOOKAHEAD] - prices[i]) / prices[i]
        threshold = (ATR_LABEL_MULT * atr_i) / prices[i]

        if ret >= threshold:
            label = 1
        elif ret <= -threshold:
            label = -1
        else:
            label = 0

        labels_aligned[i] = label   # simpan untuk candle_model
        row = _score_at(df, i)
        row["label"] = label
        row["price"] = float(prices[i])
        rows.append(row)

    # Guard: rows kosong → return DataFrame kosong dengan kolom yang benar
    if not rows:
        import logging as _log
        _log.getLogger(__name__).warning(
            "[trainer] _build_feature_matrix: 0 rows terbentuk — "
            "semua candle di-skip (ATR NaN atau LOOKAHEAD)."
        )
        empty = pd.DataFrame(columns=FEATURES_CORE + ["label", "price"])
        return empty, labels_aligned

    result = pd.DataFrame(rows)

    # Guard: kalau kolom tidak ada
    missing = [c for c in FEATURES_CORE if c not in result.columns]
    if missing:
        import logging as _log
        _log.getLogger(__name__).error(
            "[trainer] feat_df missing columns: %s. Tersedia: %s",
            missing, list(result.columns)
        )
        empty = pd.DataFrame(columns=FEATURES_CORE + ["label", "price"])
        return empty, labels_aligned

    # Buang baris yang semua indikator nol (candle terlalu awal, window < 210)
    result = result[result[FEATURES_CORE].any(axis=1)].copy()
    result["label"] = result["label"].astype(int)
    return result.reset_index(drop=True), labels_aligned


# ------------------------------------------------------------------
# Undersample class dominan — lebih valid dari inject sintetis
# ------------------------------------------------------------------

def _balance_classes(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Undersample class yang terlalu dominan.
    Jika rasio class terbesar / class terkecil > MAX_CLASS_RATIO,
    potong class dominan ke MAX_CLASS_RATIO × ukuran class terkecil.

    Lebih honest daripada inject sintetis: tidak ada data yang
    "diciptakan", hanya data yang kurang representatif dikurangi.
    """
    counts   = train_df["label"].value_counts()
    min_size = int(counts.min())
    max_allowed = int(min_size * MAX_CLASS_RATIO)

    parts = []
    for label_val, count in counts.items():
        subset = train_df[train_df["label"] == label_val]
        if count > max_allowed:
            subset = subset.sample(n=max_allowed, random_state=42)
            logger.info(
                "[trainer] Undersample class %d: %d → %d rows",
                label_val, count, max_allowed,
            )
        parts.append(subset)

    balanced = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced


# ------------------------------------------------------------------
# Pastikan semua class ada di y_train — tanpa inject sintetis
# ------------------------------------------------------------------

def _check_classes(feat_df: pd.DataFrame, symbol: str) -> bool:
    present = set(feat_df["label"].unique())
    needed  = {-1, 0, 1}
    missing = needed - present
    if missing:
        logger.warning(
            "[trainer] %s — class hilang setelah filtering: %s. "
            "Data mungkin terlalu sedikit atau pasar satu arah terus. Skip training.",
            symbol, sorted(missing),
        )
        return False
    return True


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

    Perubahan vs versi lama:
      - Hanya FEATURES_CORE (7 indikator) yang jadi fitur ML
      - Label berdasarkan ATR, bukan flat percentage
      - Class imbalance ditangani via undersample + class_weight, bukan inject sintetis
      - Funding & LSR tidak ada leakage karena tidak dipakai sebagai fitur rolling
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return {"ok": False, "reason": "XGBoost belum terinstall. Jalankan: pip install xgboost"}

    symbol = symbol.upper()

    # -- Fetch kline --
    logger.info("[trainer] Fetch %s candle %s %s", limit, symbol, interval)
    raw_df    = get_df(symbol, interval=interval, limit=limit)
    n_candles = len(raw_df)

    if n_candles < MIN_CANDLE_TRAIN:
        return {
            "ok":     False,
            "reason": f"Data tidak cukup: {n_candles} candle (minimal {MIN_CANDLE_TRAIN})",
            "symbol": symbol,
        }

    # -- Fetch funding & lsr untuk PREDICTOR (bukan untuk training features) --
    logger.info("[trainer] Fetch funding rate & LSR untuk %s (untuk predictor saja)...", symbol)
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

    # -- Build feature matrix (ATR-based labeling, no leakage) --
    logger.info("[trainer] Building feature matrix (%d candles, ATR-based labels)...", n_candles)
    feat_df, labels_aligned = _build_feature_matrix(raw_df)

    if len(feat_df) < 50:
        return {
            "ok":     False,
            "reason": f"Feature matrix terlalu kecil: {len(feat_df)} baris valid",
            "symbol": symbol,
        }

    # -- Log distribusi label --
    label_counts = feat_df["label"].value_counts().to_dict()
    logger.info("[trainer] Label distribution untuk %s: %s", symbol, label_counts)

    # -- Split train/test SEBELUM balancing --
    split_idx = int(len(feat_df) * 0.70)
    train_df  = feat_df.iloc[:split_idx].copy()
    test_df   = feat_df.iloc[split_idx:].copy()

    # -- Cek class completeness --
    if not _check_classes(train_df, symbol):
        return {
            "ok":     False,
            "reason": f"Class tidak lengkap di training set untuk {symbol}. "
                      f"Distribution: {label_counts}",
            "symbol": symbol,
        }

    # -- Balance hanya train set via undersample --
    train_df = _balance_classes(train_df)
    balanced_counts = train_df["label"].value_counts().to_dict()
    logger.info("[trainer] Balanced train distribution untuk %s: %s", symbol, balanced_counts)

    X_train = train_df[FEATURES_CORE].astype(float).values
    y_train = train_df["label"].values + 1   # shift -1,0,1 → 0,1,2
    X_test  = test_df[FEATURES_CORE].astype(float).values
    y_test  = test_df["label"].values + 1

    # -- Train dengan class_weight balanced sebagai safety net --
    logger.info(
        "[trainer] Training XGBoost (%d train / %d test) untuk %s...",
        len(X_train), len(X_test), symbol,
    )

    # Hitung sample_weight — class terkecil dapat bobot tertinggi
    class_counts = np.bincount(y_train, minlength=3)
    total        = len(y_train)
    class_w      = {c: total / (3.0 * max(class_counts[c], 1)) for c in range(3)}
    sample_weight = np.array([class_w[y] for y in y_train])

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
        sample_weight=sample_weight,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # -- Feature importance → bobot --
    raw_imp     = model.feature_importances_
    importances = {FEATURES_CORE[i]: float(raw_imp[i]) for i in range(len(FEATURES_CORE))}
    mean_imp    = float(np.mean(raw_imp))

    if mean_imp > 0:
        weights_after = {f: round(float(importances[f]) / mean_imp, 6) for f in FEATURES_CORE}
    else:
        weights_after = {f: DEFAULT_WEIGHTS[f] for f in FEATURES_CORE}

    weights_before = load_weights(symbol)
    save_weights(symbol, weights_after)
    logger.info("[trainer] Weights saved for %s (core features only)", symbol)

    # -- Train ML 2: Candle Model --
    candle_result = None
    try:
        from ml.candle_model import train_candle_model, backtest_candle_model
        logger.info("[trainer] Training candle model untuk %s...", symbol)
        candle_result = train_candle_model(symbol, raw_df, labels_aligned)
        if candle_result["ok"]:
            candle_bt = backtest_candle_model(candle_result)
            candle_result["backtest"] = candle_bt
            logger.info(
                "[trainer] Candle model %s — acc=%.1f%% wr_up=%.1f%% wr_dn=%.1f%%",
                symbol,
                candle_bt["accuracy"]   * 100,
                candle_bt["winrate_up"] * 100,
                candle_bt["winrate_dn"] * 100,
            )
        else:
            logger.warning("[trainer] Candle model gagal untuk %s: %s",
                           symbol, candle_result.get("reason"))
    except Exception as e:
        logger.warning("[trainer] Candle model error untuk %s: %s", symbol, e)
        candle_result = None

    return {
        "ok":              True,
        "symbol":          symbol,
        "interval":        interval,
        "n_candles":       n_candles,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "label_counts":    label_counts,
        "balanced_counts": balanced_counts,
        "had_missing_class": False,
        "importances":     importances,
        "weights_before":  weights_before,
        "weights_after":   weights_after,
        "feature_df":      feat_df,
        "raw_df":          raw_df,
        "fund_df":         fund_df,
        "lsr_df":          lsr_df,
        "model":           model,           # ML 1: scoring model
        "candle_result":   candle_result,   # ML 2: candle model (None jika gagal)
        "labels_aligned":  labels_aligned,  # label per candle untuk referensi
        "atr_label_mult":  ATR_LABEL_MULT,
    }
