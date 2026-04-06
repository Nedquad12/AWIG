"""
ml/regime_detector.py — Klasifikasi regime market per candle/fold.

Tiga sinyal:
  1. ADX(14)               → trending strength
  2. ATR percentile(100)   → volatilitas relatif vs sejarah
  3. Return autocorr(20,1) → trending vs mean-reverting

Priority: Volatile > Trending > Sideways

Setiap regime punya parameter model sendiri (ATR_LABEL_MULT, kelly_mult, dll).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Regime names ──────────────────────────────────────────────────────
TRENDING  = "Trending"
SIDEWAYS  = "Sideways"
VOLATILE  = "Volatile"

# ── ADX thresholds ────────────────────────────────────────────────────
ADX_PERIOD      = 14
ADX_TREND_MIN   = 25     # ADX > 25 → trending
ADX_SIDEWAYS_MAX= 20     # ADX < 20 → sideways pasti

# ── ATR percentile thresholds ─────────────────────────────────────────
ATR_PERIOD      = 14
ATR_LOOKBACK    = 100    # lookback untuk percentile
ATR_HIGH_PCT    = 75     # ATR sekarang > P75 → volatile
ATR_LOW_PCT     = 25     # ATR sekarang < P25 → low vol (sideways support)

# ── Autocorrelation thresholds ────────────────────────────────────────
AUTOCORR_LOOKBACK = 20
AUTOCORR_LAG      = 1
AUTOCORR_TREND    = +0.10   # > +0.10 → trending (momentum)
AUTOCORR_MR       = -0.10   # < -0.10 → mean-reverting / sideways

# ── Parameter per regime ──────────────────────────────────────────────
REGIME_PARAMS = {
    TRENDING: {
        "atr_label_mult":    0.5,  # turun dari 1.5 → lebih banyak UP/DOWN label
        "kelly_multiplier":  0.20,
        "min_signal_sample": 20,
        "feature_bias":      ["ma", "macd"],    # indikator yang lebih relevan
        "description": "Directional momentum. MA + MACD lebih reliable.",
    },
    SIDEWAYS: {
        "atr_label_mult":    0.3,  # turun dari 1.0
        "kelly_multiplier":  0.15,   # edge lebih tipis di sideways
        "min_signal_sample": 20,
        "feature_bias":      ["rsi", "wcc"],
        "description": "Range-bound. RSI + WCC mean-reversion lebih reliable.",
    },
    VOLATILE: {
        "atr_label_mult":    0.7,  # turun dari 2.0
        "kelly_multiplier":  0.12,   # uncertainty tinggi
        "min_signal_sample": 30,     # butuh lebih banyak sampel
        "feature_bias":      ["vsa", "fsa"],
        "description": "Unusual volatility. VSA + FSA volume-based lebih reaktif.",
    },
}


# ------------------------------------------------------------------
# ADX (Average Directional Index)
# ------------------------------------------------------------------

def _compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    """
    Hitung ADX dari candle terakhir.
    Pakai True Range dan Directional Movement standard.
    Return nilai ADX float (0–100).
    """
    if len(df) < period * 2:
        return 20.0   # default netral jika data tidak cukup

    high   = df["high"].values
    low    = df["low"].values
    close  = df["close"].values
    n      = len(df)

    tr    = np.zeros(n)
    dm_p  = np.zeros(n)
    dm_m  = np.zeros(n)

    for i in range(1, n):
        hl  = high[i]  - low[i]
        hpc = abs(high[i]  - close[i - 1])
        lpc = abs(low[i]   - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up_move   = high[i]  - high[i - 1]
        down_move = low[i - 1] - low[i]

        dm_p[i] = up_move   if (up_move > down_move and up_move > 0)   else 0.0
        dm_m[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    def _smooth(arr, p):
        result = np.zeros(n)
        result[p] = arr[1:p + 1].sum()
        for i in range(p + 1, n):
            result[i] = result[i - 1] - result[i - 1] / p + arr[i]
        return result

    atr_s  = _smooth(tr,   period)
    dmp_s  = _smooth(dm_p, period)
    dmm_s  = _smooth(dm_m, period)

    di_p = np.where(atr_s > 0, 100 * dmp_s / atr_s, 0)
    di_m = np.where(atr_s > 0, 100 * dmm_s / atr_s, 0)
    dx   = np.where((di_p + di_m) > 0, 100 * np.abs(di_p - di_m) / (di_p + di_m), 0)

    adx  = np.zeros(n)
    adx[period * 2 - 1] = dx[period: period * 2].mean()
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return float(adx[-1])


# ------------------------------------------------------------------
# ATR percentile
# ------------------------------------------------------------------

def _compute_atr_percentile(df: pd.DataFrame, period: int = ATR_PERIOD, lookback: int = ATR_LOOKBACK) -> float:
    """
    Hitung ATR sekarang sebagai percentile dari ATR di lookback candle terakhir.
    Return 0–100 (percentile rank).
    """
    if len(df) < lookback + period:
        return 50.0   # default netral

    high   = df["high"].values
    low    = df["low"].values
    close  = df["close"].values
    n      = len(df)

    # Rolling ATR
    tr_arr = np.zeros(n)
    for i in range(1, n):
        hl  = high[i] - low[i]
        hpc = abs(high[i]  - close[i - 1])
        lpc = abs(low[i]   - close[i - 1])
        tr_arr[i] = max(hl, hpc, lpc)

    atr_arr = np.full(n, np.nan)
    for i in range(period, n):
        atr_arr[i] = tr_arr[i - period + 1: i + 1].mean()

    # Ambil lookback candle terakhir yang valid
    valid = atr_arr[~np.isnan(atr_arr)]
    if len(valid) < 2:
        return 50.0

    history = valid[-lookback:]
    current = valid[-1]

    pct_rank = float(np.mean(history <= current)) * 100
    return round(pct_rank, 2)


# ------------------------------------------------------------------
# Return autocorrelation
# ------------------------------------------------------------------

def _compute_autocorr(df: pd.DataFrame, lookback: int = AUTOCORR_LOOKBACK, lag: int = AUTOCORR_LAG) -> float:
    """
    Hitung autocorrelation return dengan lag=1 dari lookback candle terakhir.
    Return -1 sampai +1.
    """
    if len(df) < lookback + lag + 1:
        return 0.0   # default noise

    close   = df["close"].values[-lookback - lag:]
    returns = np.diff(close) / close[:-1]

    if len(returns) < lookback:
        return 0.0

    r     = returns[-lookback:]
    r_lag = returns[-lookback - lag: -lag] if lag > 0 else r

    if len(r) != len(r_lag):
        return 0.0

    if np.std(r) < 1e-10 or np.std(r_lag) < 1e-10:
        return 0.0

    corr = float(np.corrcoef(r, r_lag)[0, 1])
    return round(corr if not np.isnan(corr) else 0.0, 4)


# ------------------------------------------------------------------
# Classify regime
# ------------------------------------------------------------------

def classify(
    adx: float,
    atr_pct: float,
    autocorr: float,
) -> str:
    """
    Decision matrix:
      Priority: Volatile > Trending > Sideways

      Volatile  : ATR percentile > P75 (unusually high vol)
      Trending  : ADX > 25 AND ATR normal AND autocorr > -0.10
      Sideways  : ADX < 20 OR (ADX 20–25 AND autocorr < +0.10)
    """
    # Priority 1: Volatile (override semua kondisi lain)
    if atr_pct >= ATR_HIGH_PCT:
        return VOLATILE

    # Priority 2: Trending
    if adx >= ADX_TREND_MIN and autocorr >= AUTOCORR_MR:
        return TRENDING

    # Priority 3: Sideways
    if adx < ADX_SIDEWAYS_MAX:
        return SIDEWAYS

    # ADX 20–25 (transisi)
    if autocorr >= AUTOCORR_TREND:
        return TRENDING
    return SIDEWAYS


# ------------------------------------------------------------------
# Public: detect_regime
# ------------------------------------------------------------------

def detect_regime(df: pd.DataFrame) -> dict:
    """
    Deteksi regime dari DataFrame candle.
    Biasanya dipanggil dengan training window atau OOS window.

    Returns dict:
        regime      : "Trending" | "Sideways" | "Volatile"
        adx         : nilai ADX
        atr_pct     : ATR percentile (0–100)
        autocorr    : return autocorrelation lag-1
        params      : REGIME_PARAMS[regime]
        description : deskripsi singkat
    """
    adx      = _compute_adx(df)
    atr_pct  = _compute_atr_percentile(df)
    autocorr = _compute_autocorr(df)
    regime   = classify(adx, atr_pct, autocorr)

    logger.info(
        "[regime] Detected: %s | ADX=%.1f | ATR_pct=%.0f%% | autocorr=%.3f",
        regime, adx, atr_pct, autocorr,
    )

    return {
        "regime":      regime,
        "adx":         round(adx,     2),
        "atr_pct":     round(atr_pct, 1),
        "autocorr":    autocorr,
        "params":      REGIME_PARAMS[regime],
        "description": REGIME_PARAMS[regime]["description"],
    }


# ------------------------------------------------------------------
# Format regime untuk prompt AI
# ------------------------------------------------------------------

def format_for_ai(regime_info: dict, wfv_history: list[dict]) -> str:
    """
    Format regime + WFV history untuk dikasih ke AI sebagai konteks.

    wfv_history: list of fold results dari wfv.py
    """
    r      = regime_info
    params = r["params"]

    lines = [
        f"=== REGIME SAAT INI ===",
        f"  Regime   : {r['regime']}",
        f"  ADX      : {r['adx']:.1f} (trend strength, >25 = trending)",
        f"  ATR pct  : {r['atr_pct']:.0f}% (volatilitas relatif, >75% = unusual)",
        f"  Autocorr : {r['autocorr']:+.3f} (>+0.10 = momentum, <-0.10 = mean-revert)",
        f"  Desc     : {r['description']}",
        f"  Model    : ATR_MULT={params['atr_label_mult']} | Kelly={params['kelly_multiplier']*100:.0f}% full Kelly",
        f"",
        f"=== WFV HISTORY PER REGIME (modal $100 simulasi) ===",
    ]

    if not wfv_history:
        lines.append("  Belum ada WFV history.")
        return "\n".join(lines)

    # Agregat per regime
    from collections import defaultdict
    regime_stats: dict[str, dict] = defaultdict(lambda: {
        "folds": 0, "profitable_folds": 0,
        "total_net_pnl": 0.0, "total_trades": 0,
        "total_wins": 0,
    })

    for fold in wfv_history:
        reg = fold.get("regime", "Unknown")
        s   = regime_stats[reg]
        s["folds"]           += 1
        s["total_net_pnl"]   += fold.get("net_pnl", 0.0)
        s["total_trades"]    += fold.get("n_trades", 0)
        s["total_wins"]      += fold.get("n_wins",   0)
        if fold.get("net_pnl", 0) > 0:
            s["profitable_folds"] += 1

    for reg, s in sorted(regime_stats.items()):
        n_trades = s["total_trades"]
        wr       = (s["total_wins"] / n_trades * 100) if n_trades > 0 else 0.0
        pnl_sign = "+" if s["total_net_pnl"] >= 0 else ""
        lines.append(
            f"  {reg:<10} → "
            f"WR: {wr:.1f}% | "
            f"PnL net: {pnl_sign}${s['total_net_pnl']:.2f} | "
            f"{n_trades} trades | "
            f"profitable {s['profitable_folds']}/{s['folds']} fold"
        )

    # Highlight regime sekarang
    current = regime_info["regime"]
    if current in regime_stats:
        s    = regime_stats[current]
        prof = s["profitable_folds"]
        tot  = s["folds"]
        pnl  = s["total_net_pnl"]
        if tot > 0 and (prof / tot < 0.5 or pnl < 0):
            lines.append(
                f"\n  ⚠️  Regime saat ini ({current}) historically "
                f"unprofitable ({prof}/{tot} fold profit, net ${pnl:.2f}). "
                f"Pertimbangkan SKIP."
            )

    return "\n".join(lines)
