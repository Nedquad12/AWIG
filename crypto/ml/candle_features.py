"""
ml/candle_features.py — Raw price feature engineering + regime detection.

Dua tanggung jawab:
  1. build_candle_features(df, n=20)
     Ekstrak fitur dari 20 candle terakhir per titik waktu.
     Fitur ini TIDAK lagging — langsung dari price action.

  2. detect_regime(df) → "TRENDING" | "SIDEWAYS"
     Pakai ADX (Average Directional Index).
     ADX > 25 → TRENDING  → sinyal momentum reliable
     ADX < 20 → SIDEWAYS  → sinyal sering false, lebih konservatif

Fitur yang diekstrak (per candle window):
  Momentum  : return_1, return_3, return_5, return_10
  Volatility: vol_10, vol_20 (realized std of returns)
  Volume    : vol_ratio_7_30 (spike detection)
  Structure : hl_range_pct, close_pos, upper_wick, lower_wick
  Trend     : trend_slope_20, mean_rev_z (z-score vs MA20)
  Regime    : adx_14 (sebagai fitur numerik, bukan hanya label)
"""

import numpy as np
import pandas as pd

# Jumlah candle terakhir yang dipakai sebagai fitur
N_CANDLES = 20

# ADX thresholds
ADX_TRENDING  = 25.0
ADX_SIDEWAYS  = 20.0
ADX_PERIOD    = 14


# ------------------------------------------------------------------
# ADX Calculation
# ------------------------------------------------------------------

def _compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> np.ndarray:
    """
    Hitung ADX untuk setiap baris secara rolling.
    Return array float, NaN untuk baris yang belum cukup data.

    ADX mengukur KEKUATAN trend (bukan arah).
    Tidak ada lookahead — adx[i] hanya pakai data sampai candle i.
    """
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr       = np.zeros(n)

    for i in range(1, n):
        up   = highs[i]  - highs[i - 1]
        down = lows[i - 1] - lows[i]

        plus_dm[i]  = up   if (up > down and up > 0)   else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

        hl  = highs[i] - lows[i]
        hpc = abs(highs[i]  - closes[i - 1])
        lpc = abs(lows[i]   - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    # Wilder smoothing
    def _wilder(arr, p):
        out = np.full(n, np.nan)
        if n <= p:
            return out
        out[p] = np.sum(arr[1: p + 1])
        for i in range(p + 1, n):
            out[i] = out[i - 1] - (out[i - 1] / p) + arr[i]
        return out

    atr14    = _wilder(tr,       period)
    plus14   = _wilder(plus_dm,  period)
    minus14  = _wilder(minus_dm, period)

    plus_di  = np.where(atr14 > 0, 100 * plus14  / atr14, 0.0)
    minus_di = np.where(atr14 > 0, 100 * minus14 / atr14, 0.0)
    di_sum   = plus_di + minus_di
    dx       = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    adx = np.full(n, np.nan)
    start = period * 2
    if n > start:
        adx[start] = np.nanmean(dx[period: start + 1])
        for i in range(start + 1, n):
            if not np.isnan(adx[i - 1]):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


# ------------------------------------------------------------------
# Single-row candle features (untuk satu candle i)
# ------------------------------------------------------------------

def _features_at(
    df: pd.DataFrame,
    i: int,
    adx_arr: np.ndarray,
    n: int = N_CANDLES,
) -> dict[str, float] | None:
    """
    Ekstrak fitur dari window [i-n+1 : i+1] (n candle terakhir sampai candle i).
    Return None jika data tidak cukup.
    """
    if i < n + 9:   # butuh n candle window + 10 untuk vol_30
        return None

    closes  = df["close"].values.astype(float)
    highs   = df["high"].values.astype(float)
    lows    = df["low"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    # Window closes untuk n candle
    w_close = closes[i - n + 1: i + 1]   # shape (n,)
    w_high  = highs[i  - n + 1: i + 1]
    w_low   = lows[i   - n + 1: i + 1]
    w_vol   = volumes[i - n + 1: i + 1]

    c  = closes[i]
    if c == 0:
        return None

    # --- Momentum: return dari berbagai lookback ---
    def _ret(lb):
        idx = i - lb
        if idx < 0 or closes[idx] == 0:
            return 0.0
        return (c - closes[idx]) / closes[idx]

    ret_1  = _ret(1)
    ret_3  = _ret(3)
    ret_5  = _ret(5)
    ret_10 = _ret(10)

    # --- Volatility: realized std of 1-period returns ---
    rets_w = np.diff(w_close) / np.where(w_close[:-1] != 0, w_close[:-1], 1.0)
    vol_10 = float(np.std(rets_w[-10:])) if len(rets_w) >= 10 else 0.0
    vol_20 = float(np.std(rets_w))       if len(rets_w) >= 2  else 0.0

    # --- Volume ratio: avg 7 vs avg 30 ---
    vol_7  = float(np.mean(volumes[i - 6:  i + 1]))  if i >= 6  else 0.0
    vol_30 = float(np.mean(volumes[i - 29: i + 1]))  if i >= 29 else vol_7
    vol_ratio = vol_7 / vol_30 if vol_30 > 0 else 1.0

    # --- Candle structure dari candle terbaru ---
    o   = float(df["open"].values[i])
    h   = float(highs[i])
    l   = float(lows[i])
    hl  = h - l if h != l else 1e-10

    hl_range_pct  = hl / c                           # range relatif terhadap harga
    close_pos     = (c - l) / hl                     # 0=bottom, 1=top of range
    upper_wick    = (h - max(o, c)) / hl             # upper wick ratio
    lower_wick    = (min(o, c) - l) / hl             # lower wick ratio

    # --- Trend strength: slope linear regression atas 20 candle terakhir ---
    x = np.arange(n, dtype=float)
    if np.std(w_close) > 0:
        slope = float(np.polyfit(x, w_close, 1)[0])
        trend_slope = slope / c   # normalisasi vs harga
    else:
        trend_slope = 0.0

    # --- Mean reversion: z-score vs MA20 ---
    ma20 = float(np.mean(w_close))
    std20 = float(np.std(w_close))
    mean_rev_z = (c - ma20) / std20 if std20 > 0 else 0.0

    # --- ADX sebagai fitur numerik ---
    adx_val = float(adx_arr[i]) if not np.isnan(adx_arr[i]) else 0.0

    return {
        # Momentum
        "ret_1":       float(ret_1),
        "ret_3":       float(ret_3),
        "ret_5":       float(ret_5),
        "ret_10":      float(ret_10),
        # Volatility
        "vol_10":      float(vol_10),
        "vol_20":      float(vol_20),
        # Volume
        "vol_ratio":   float(vol_ratio),
        # Structure
        "hl_range":    float(hl_range_pct),
        "close_pos":   float(close_pos),
        "upper_wick":  float(upper_wick),
        "lower_wick":  float(lower_wick),
        # Trend
        "trend_slope": float(trend_slope),
        "mean_rev_z":  float(mean_rev_z),
        # Regime (numerik)
        "adx":         float(adx_val),
    }


# ------------------------------------------------------------------
# Nama fitur (harus konsisten di semua file)
# ------------------------------------------------------------------

CANDLE_FEATURE_NAMES: list[str] = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "vol_10", "vol_20",
    "vol_ratio",
    "hl_range", "close_pos", "upper_wick", "lower_wick",
    "trend_slope", "mean_rev_z",
    "adx",
]


# ------------------------------------------------------------------
# Build full feature matrix (rolling, semua candle)
# ------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Build DataFrame fitur candle untuk training.

    Args:
        df     : raw OHLCV DataFrame, ascending
        labels : array label (-1, 0, 1) per candle — harus sama panjang dengan df

    Returns:
        DataFrame dengan kolom CANDLE_FEATURE_NAMES + 'label' + 'price'
        Hanya baris yang fiturnya valid (tidak None).
    """
    assert len(df) == len(labels), "df dan labels harus sama panjang"

    adx_arr = _compute_adx(df)
    rows    = []

    for i in range(len(df)):
        feat = _features_at(df, i, adx_arr)
        if feat is None:
            continue
        if labels[i] is None or np.isnan(float(labels[i])):
            continue

        feat["label"] = int(labels[i])
        feat["price"] = float(df["close"].values[i])
        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=CANDLE_FEATURE_NAMES + ["label", "price"])

    result = pd.DataFrame(rows)
    result["label"] = result["label"].astype(int)
    return result.reset_index(drop=True)


# ------------------------------------------------------------------
# Fitur untuk satu prediksi (candle terakhir saja)
# ------------------------------------------------------------------

def get_current_features(df: pd.DataFrame) -> dict[str, float] | None:
    """
    Ekstrak fitur dari candle terakhir untuk dipakai di predictor.
    Return None jika data tidak cukup.
    """
    adx_arr = _compute_adx(df)
    return _features_at(df, len(df) - 1, adx_arr)


# ------------------------------------------------------------------
# Regime detection
# ------------------------------------------------------------------

def detect_regime(df: pd.DataFrame) -> str:
    """
    Deteksi regime pasar dari candle terakhir.

    Returns:
        "TRENDING"  — ADX > 25, sinyal momentum reliable
        "SIDEWAYS"  — ADX < 20, sinyal sering false
        "NEUTRAL"   — ADX antara 20-25, transisi
    """
    adx_arr = _compute_adx(df)

    # Ambil ADX dari beberapa candle terakhir untuk smoothing
    recent = adx_arr[-5:]
    valid  = recent[~np.isnan(recent)]

    if len(valid) == 0:
        return "NEUTRAL"

    adx_val = float(np.mean(valid))

    if adx_val > ADX_TRENDING:
        return "TRENDING"
    elif adx_val < ADX_SIDEWAYS:
        return "SIDEWAYS"
    else:
        return "NEUTRAL"


def get_regime_weight(regime: str) -> float:
    """
    Return multiplier untuk scoring model berdasarkan regime.

    TRENDING  → scoring model lebih reliable (indikator momentum bekerja)
    SIDEWAYS  → scoring model kurang reliable (false signals banyak)
    NEUTRAL   → intermediate

    Ini dipakai di predictor untuk adjust bobot scoring model relatif
    terhadap candle model. Candle model tidak dipengaruhi regime weight
    karena fiturnya sudah include ADX sebagai fitur.
    """
    return {
        "TRENDING": 1.0,    # full weight
        "NEUTRAL":  0.7,    # sedikit dikurangi
        "SIDEWAYS": 0.4,    # dikurangi signifikan
    }.get(regime, 0.7)
