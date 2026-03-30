# =============================================================
# scanner.py — Scan top N token, hitung semua 9 indikator
# =============================================================

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

sys.path.append(os.path.dirname(__file__))
from config import (
    BINANCE_BASE_URL, DEFAULT_INTERVAL,
    SCAN_SCORE_THRESHOLD, SCAN_TOP_N,
    SCANNER_MAX_WORKERS, SCANNER_REQUEST_DELAY,
    SCANNER_BLACKLIST,
)
from indicators.binance_fetcher import get_df
from indicators import (
    score_vsa, score_fsa, score_vfa,
    score_rsi, score_macd, score_ma, score_wcc,
)
from indicators.funding import fetch_funding_rate, score_funding, get_funding_detail
from indicators.lsr     import fetch_lsr, score_lsr, get_lsr_detail
from ml.weight_manager  import load_weights, apply_weights

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# BTC Multiplier
# ------------------------------------------------------------------

def get_btc_multiplier(interval: str = DEFAULT_INTERVAL) -> tuple[float, float]:
    """
    Scan BTCUSDT duluan untuk market context, return (btc_weighted_total, signed_multiplier).

    Multiplier table (berdasarkan abs weighted_total):
      < 1.0      →  0.0  (netral, no adjustment)
      1.0 – 3.0  →  1.0
      3.0 – 6.0  →  2.0
      6.0 – 8.0  →  2.5
      8.0+       →  3.5

    Sign multiplier ikut sign BTC weighted_total:
      BTC positif → multiplier positif (boost bull, lemahkan bear)
      BTC negatif → multiplier negatif (boost bear, lemahkan bull)
      BTC netral  → multiplier 0.0 (no adjustment)

    Returns:
        (btc_weighted_total, signed_multiplier)
    """
    logger.info("[scanner] Scanning BTCUSDT untuk market context...")
    result = score_symbol("BTCUSDT", interval=interval)

    if result is None:
        logger.warning("[scanner] BTC scan gagal, multiplier default 0.0")
        return 0.0, 0.0

    btc_total = result["weighted_total"]
    abs_total = abs(btc_total)

    if abs_total < 1.0:
        raw_mult = 0.0
    elif abs_total < 3.0:
        raw_mult = 1.0
    elif abs_total < 6.0:
        raw_mult = 2.0
    elif abs_total < 8.0:
        raw_mult = 2.5
    else:
        raw_mult = 3.5

    # Sign ikut BTC
    if btc_total > 0:
        signed_mult = raw_mult
    elif btc_total < 0:
        signed_mult = -raw_mult
    else:
        signed_mult = 0.0

    logger.info(
        "[scanner] BTC weighted_total=%.4f → multiplier=%+.1f",
        btc_total, signed_mult,
    )
    return btc_total, signed_mult


def apply_btc_multiplier(weighted_total: float, btc_multiplier: float) -> float:
    """
    Apply BTC multiplier ke weighted_total altcoin.

    Logic:
      - BTC multiplier 0.0 (netral) → weighted_total tidak berubah
      - Arah altcoin SAMA dengan BTC → dikali abs_multiplier (dikuatkan)
      - Arah altcoin BERLAWANAN dengan BTC → dibagi abs_multiplier (dilemahkan)

    Contoh:
      altcoin = +3.0, BTC mult = +2.0 (bullish) → +3.0 × 2.0 = +6.0  (sama arah, dikuat)
      altcoin = -2.0, BTC mult = +2.0 (bullish) → -2.0 / 2.0 = -1.0  (lawan arah, dilemah)
      altcoin = +3.0, BTC mult = -2.0 (bearish) → +3.0 / 2.0 = +1.5  (lawan arah, dilemah)
      altcoin = -2.0, BTC mult = -2.0 (bearish) → -2.0 × 2.0 = -4.0  (sama arah, dikuat)
      altcoin = +3.0, BTC mult =  0.0 (netral)  → +3.0         (no change)
    """
    if btc_multiplier == 0.0:
        return weighted_total

    abs_mult = abs(btc_multiplier)

    # Arah sama → kuatkan
    if (btc_multiplier > 0 and weighted_total > 0) or \
       (btc_multiplier < 0 and weighted_total < 0):
        return round(weighted_total * abs_mult, 4)

    # Arah berlawanan → lemahkan
    return round(weighted_total / abs_mult, 4)


# ------------------------------------------------------------------
# Step 1: Ambil top N symbol by 24h quote volume
# ------------------------------------------------------------------

def get_top_symbols(top_n: int = SCAN_TOP_N) -> list[str]:
    """
    Return list symbol USDT perpetual futures sorted by 24h quoteVolume.

    Filter:
      - Hanya simbol yang endswith 'USDT' (otomatis exclude ETHBTC, BNBETH, dll)
      - Exclude semua simbol di SCANNER_BLACKLIST (BTCUSDT, stablecoin pair, dll)
    """
    url  = f"{BINANCE_BASE_URL}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    tickers = resp.json()

    usdt_perps = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in SCANNER_BLACKLIST
    ]
    usdt_perps.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    symbols = [t["symbol"] for t in usdt_perps[:top_n]]
    logger.info("[scanner] Top %d symbols (blacklist excluded): %s...", top_n, symbols[:5])
    return symbols


# ------------------------------------------------------------------
# Step 2: Scoring satu token (semua 9 indikator)
# ------------------------------------------------------------------

def score_symbol(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    kline_limit: int = 210,
) -> Optional[dict]:
    """
    Hitung semua 9 indikator untuk satu symbol.
    Retry 3x dengan exponential backoff jika gagal.
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # 1. Kline data (untuk 7 indikator utama)
            df = get_df(symbol, interval=interval, limit=kline_limit)
            if df is None or len(df) < 50:
                logger.warning("[scanner] %s: data kline tidak cukup (%s baris)",
                               symbol, len(df) if df is not None else 0)
                return None

            scores = {
                "vsa":  float(score_vsa(df)),
                "fsa":  float(score_fsa(df)),
                "vfa":  float(score_vfa(df)),
                "rsi":  float(score_rsi(df)),
                "macd": float(score_macd(df)),
                "ma":   float(score_ma(df)),
                "wcc":  float(score_wcc(df)),
            }

            # 2. Funding rate
            time.sleep(SCANNER_REQUEST_DELAY)
            fund_df = fetch_funding_rate(symbol, limit=90)
            scores["funding"] = float(score_funding(fund_df))
            funding_detail    = get_funding_detail(fund_df)

            # 3. Long/Short Ratio
            time.sleep(SCANNER_REQUEST_DELAY)
            lsr_df = fetch_lsr(symbol, interval=interval, limit=96)
            scores["lsr"]  = float(score_lsr(lsr_df))
            lsr_detail     = get_lsr_detail(lsr_df)

            # 4. Weighted total (raw, sebelum BTC multiplier)
            weights        = load_weights(symbol)
            weighted_total = apply_weights(scores, weights)

            direction = (
                "LONG"    if weighted_total > 0 else
                "SHORT"   if weighted_total < 0 else
                "NEUTRAL"
            )

            return {
                "symbol":         symbol,
                "scores":         scores,
                "weighted_total": round(weighted_total, 4),
                "direction":      direction,
                "funding_detail": funding_detail,
                "lsr_detail":     lsr_detail,
                "raw_df":         df,
            }

        except requests.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 429 or status == 418:
                wait = 10 * (attempt + 1)
                logger.warning("[scanner] %s rate limited (HTTP %d) — tunggu %ds", symbol, status, wait)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                logger.warning("[scanner] %s HTTP %d — retry %ds", symbol, status, wait)
                time.sleep(wait)
            else:
                logger.warning("[scanner] %s gagal setelah %d attempt: %s", symbol, max_retries, e)
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 * (attempt + 1)
                logger.warning("[scanner] %s attempt %d gagal: %s — retry %ds",
                               symbol, attempt + 1, e, wait)
                time.sleep(wait)
            else:
                logger.warning("[scanner] %s scoring failed: %s", symbol, e)
                return None

    return None


# ------------------------------------------------------------------
# Step 3: Scan semua top symbols
# ------------------------------------------------------------------

def scan(
    top_n:     int   = SCAN_TOP_N,
    interval:  str   = DEFAULT_INTERVAL,
    threshold: float = SCAN_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Flow:
      1. Scan BTCUSDT duluan → dapat BTC multiplier
      2. Scan top_n altcoin secara paralel (BTCUSDT excluded via blacklist)
      3. Apply BTC multiplier ke weighted_total setiap altcoin
      4. Filter threshold & sort by abs(weighted_total) descending
    """
    # ── 1. BTC market context ─────────────────────────────────────
    btc_total, btc_multiplier = get_btc_multiplier(interval)

    # ── 2. Get symbols & scan paralel ─────────────────────────────
    symbols = get_top_symbols(top_n)
    logger.info(
        "[scanner] Scanning %d symbols (interval=%s, threshold=%.1f, btc_mult=%+.1f)...",
        len(symbols), interval, threshold, btc_multiplier,
    )

    results = []
    failed  = 0

    with ThreadPoolExecutor(max_workers=SCANNER_MAX_WORKERS) as executor:
        futures = {
            executor.submit(score_symbol, sym, interval): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                logger.warning("[scanner] future error %s: %s", sym, e)
                failed += 1

    # ── 3. Apply BTC multiplier ke semua altcoin ──────────────────
    for r in results:
        original = r["weighted_total"]
        adjusted = apply_btc_multiplier(original, btc_multiplier)
        r["weighted_total_raw"]     = original       # simpan nilai asli
        r["weighted_total"]         = adjusted       # nilai setelah BTC adjustment
        r["btc_multiplier_applied"] = btc_multiplier
        # Recalculate direction setelah adjustment
        r["direction"] = (
            "LONG"    if adjusted > 0 else
            "SHORT"   if adjusted < 0 else
            "NEUTRAL"
        )

    # ── 4. Filter threshold & sort ────────────────────────────────
    passed = [
        r for r in results
        if abs(r["weighted_total"]) >= threshold
        and r["direction"] != "NEUTRAL"
    ]
    passed.sort(key=lambda x: abs(x["weighted_total"]), reverse=True)

    logger.info(
        "[scanner] Done. BTC=%.4f (mult=%+.1f) | %d/%d scored | %d lolos threshold=%.1f | %d gagal.",
        btc_total, btc_multiplier, len(results), len(symbols), len(passed), threshold, failed,
    )
    return passed


# ------------------------------------------------------------------
# Format ringkasan scan untuk Telegram
# ------------------------------------------------------------------

def format_scan_summary(passed: list[dict], top_n: int, interval: str) -> str:
    if not passed:
        return (
            f"🔍 <b>Scan selesai</b> — {top_n} token ({interval})\n"
            f"⚪ Tidak ada token yang lolos threshold."
        )

    btc_mult  = passed[0].get("btc_multiplier_applied", 0.0) if passed else 0.0
    btc_label = (
        f"🟢 Bullish (×{abs(btc_mult):.1f})" if btc_mult > 0 else
        f"🔴 Bearish (×{abs(btc_mult):.1f})" if btc_mult < 0 else
        "⚪ Netral (no adjustment)"
    )

    lines = [
        f"🔍 <b>Scan Result</b> — Top {top_n} ({interval})",
        f"₿  BTC Context : {btc_label}",
        f"✅ <b>{len(passed)} token lolos</b>\n",
    ]
    for i, r in enumerate(passed[:20], 1):
        sym   = r["symbol"]
        total = r["weighted_total"]
        raw   = r.get("weighted_total_raw", total)
        dir_  = r["direction"]
        emoji = "🟢" if dir_ == "LONG" else "🔴"
        lines.append(
            f"  {i:>2}. {emoji} <b>{sym:<14}</b> "
            f"<code>{total:+.2f}</code> "
            f"<i>(raw {raw:+.2f})</i>"
        )

    if len(passed) > 20:
        lines.append(f"  ... dan {len(passed) - 20} lainnya")

    return "\n".join(lines)
