# =============================================================
# hft/wti_crypto.py — Weight-to-Index (WTI) versi Crypto
#
# Konsep sama dengan wti_command.py tapi untuk crypto:
#   - Index  = BTCUSDT (bukan IHSG/LQ45)
#   - Data   = candle 1h dari Binance Futures
#   - Window = 90 candle (≈ 3.75 hari, cukup untuk scalping)
#   - ATR threshold adaptif per koin (sama dengan WTI saham)
#
# Output:
#   wti_pct → berapa % candle koin ikut arah BTC
#   ≥ 50%   → koin "mengikuti" BTC → BTC reversal signal aktif
#   < 50%   → koin independen → hanya pakai reversal dari koin sendiri
#
# Cache:
#   Di-compute saat posisi dibuka, disimpan in-memory.
#   Tidak perlu update real-time karena korelasi berubah lambat.
# =============================================================

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"

# Window untuk hitung WTI (candle 1h)
WTI_LOOKBACK    = 90     # candle
ATR_PERIOD      = 14     # Wilder ATR
ATR_DIVISOR     = 7.0    # sama dengan wti_command.py
BTC_THRESHOLD   = 0.1    # % perubahan BTC dianggap "bergerak"
WTI_MIN_FOLLOW  = 50.0   # % minimum untuk aktifkan BTC reversal signal

# Cache: {symbol: (wti_pct, ts_computed)}
_WTI_CACHE: Dict[str, Tuple[float, float]] = {}
CACHE_TTL = 3600   # 1 jam — recompute kalau sudah basi


# ------------------------------------------------------------------
# Fetch candle dari Binance Futures
# ------------------------------------------------------------------

def _fetch_closes(symbol: str, interval: str = "1h", limit: int = 100) -> Optional[list]:
    try:
        resp = requests.get(
            f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        return [
            {
                "open":  float(c[1]),
                "high":  float(c[2]),
                "low":   float(c[3]),
                "close": float(c[4]),
            }
            for c in raw
        ]
    except Exception as e:
        logger.warning("[wti_crypto] Gagal fetch %s: %s", symbol, e)
        return None


# ------------------------------------------------------------------
# ATR Wilder (sama logika dengan wti_command.py)
# ------------------------------------------------------------------

def _calc_atr(candles: list, period: int = ATR_PERIOD) -> Optional[float]:
    if len(candles) < period + 1:
        return None

    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]
    closes = [c["close"] for c in candles]

    tr = [highs[0] - lows[0]]
    for i in range(1, len(candles)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        ))

    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period

    return atr


# ------------------------------------------------------------------
# Core WTI calculation
# ------------------------------------------------------------------

def calculate_wti_crypto(symbol: str) -> Optional[dict]:
    """
    Hitung WTI koin vs BTCUSDT dari candle 1h (90 candle terakhir).

    Returns dict dengan:
        wti_pct      → % candle koin ikut arah BTC (rata-rata up+down)
        wti_up_pct   → % BTC naik, koin ikut naik
        wti_dn_pct   → % BTC turun, koin ikut turun
        btc_active   → True jika wti_pct >= 50%
        atr_pct      → ATR% koin (volatilitas relatif)
        tkr_threshold→ threshold pergerakan koin dianggap "bergerak"
        candles_used → jumlah candle yang dipakai
    """
    symbol = symbol.upper()
    if symbol == "BTCUSDT":
        # BTC vs dirinya sendiri = 100% korelasi
        return {
            "wti_pct": 100.0, "wti_up_pct": 100.0, "wti_dn_pct": 100.0,
            "btc_active": True, "atr_pct": 0.0, "tkr_threshold": 0.0,
            "candles_used": 0,
        }

    # Fetch data
    btc_candles = _fetch_closes("BTCUSDT", limit=WTI_LOOKBACK + 5)
    tkr_candles = _fetch_closes(symbol,    limit=WTI_LOOKBACK + 5)

    if not btc_candles or not tkr_candles:
        return None

    if len(btc_candles) < 20 or len(tkr_candles) < 20:
        return None

    # ATR koin untuk threshold adaptif
    atr14     = _calc_atr(tkr_candles)
    if atr14 is None:
        return None

    last_close    = tkr_candles[-1]["close"]
    atr_pct       = (atr14 / last_close) * 100
    tkr_threshold = atr_pct / ATR_DIVISOR   # sama formula dengan WTI saham

    # Pakai candle yang overlap (keduanya punya jumlah sama dari Binance)
    n = min(len(btc_candles), len(tkr_candles), WTI_LOOKBACK)
    btc = btc_candles[-n:]
    tkr = tkr_candles[-n:]

    btc_up_total = btc_dn_total = 0
    btc_up_match = btc_dn_match = 0

    for i in range(1, n):
        btc_chg = (btc[i]["close"] - btc[i-1]["close"]) / btc[i-1]["close"] * 100
        tkr_chg = (tkr[i]["close"] - tkr[i-1]["close"]) / tkr[i-1]["close"] * 100

        btc_up = btc_chg >  BTC_THRESHOLD
        btc_dn = btc_chg < -BTC_THRESHOLD
        tkr_up = tkr_chg >  tkr_threshold
        tkr_dn = tkr_chg < -tkr_threshold

        if btc_up:
            btc_up_total += 1
            if tkr_up:
                btc_up_match += 1
        elif btc_dn:
            btc_dn_total += 1
            if tkr_dn:
                btc_dn_match += 1

    wti_up_pct = (btc_up_match / btc_up_total * 100) if btc_up_total > 0 else 0.0
    wti_dn_pct = (btc_dn_match / btc_dn_total * 100) if btc_dn_total > 0 else 0.0

    # WTI overall = rata-rata up dan down
    wti_pct    = (wti_up_pct + wti_dn_pct) / 2

    return {
        "symbol":         symbol,
        "wti_pct":        round(wti_pct, 1),
        "wti_up_pct":     round(wti_up_pct, 1),
        "wti_dn_pct":     round(wti_dn_pct, 1),
        "btc_active":     wti_pct >= WTI_MIN_FOLLOW,
        "atr_pct":        round(atr_pct, 3),
        "tkr_threshold":  round(tkr_threshold, 3),
        "candles_used":   n - 1,
        "btc_up_total":   btc_up_total,
        "btc_dn_total":   btc_dn_total,
    }


# ------------------------------------------------------------------
# Cache wrapper
# ------------------------------------------------------------------

def get_wti(symbol: str, force: bool = False) -> Optional[dict]:
    """
    Ambil WTI dari cache jika masih fresh (< 1 jam).
    Recompute jika basi atau force=True.
    """
    sym = symbol.upper()
    cached = _WTI_CACHE.get(sym)

    if cached and not force:
        wti_data, ts = cached
        if time.time() - ts < CACHE_TTL:
            return wti_data

    result = calculate_wti_crypto(sym)
    if result:
        _WTI_CACHE[sym] = (result, time.time())
        logger.info(
            "[wti_crypto] %s WTI=%.1f%% (up=%.1f%% dn=%.1f%%) btc_active=%s",
            sym, result["wti_pct"], result["wti_up_pct"],
            result["wti_dn_pct"], result["btc_active"],
        )
    return result


def is_btc_correlated(symbol: str) -> bool:
    """Shortcut: apakah koin ini mengikuti BTC (WTI ≥ 50%)?"""
    result = get_wti(symbol)
    return result["btc_active"] if result else False
