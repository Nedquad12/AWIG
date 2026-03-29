import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

sys.path.append(os.path.dirname(__file__))
from config import BINANCE_BASE_URL, DEFAULT_INTERVAL, SCAN_SCORE_THRESHOLD, SCAN_TOP_N
from indicators.binance_fetcher import get_df
from indicators import (
    score_vsa, score_fsa, score_vfa,
    score_rsi, score_macd, score_ma, score_wcc,
    INDICATOR_NAMES,
)
from indicators.funding import fetch_funding_rate, score_funding
from indicators.lsr     import fetch_lsr, score_lsr
from ml.weight_manager  import load_weights, apply_weights

logger = logging.getLogger(__name__)

# Delay antar request per token agar tidak kena rate limit
_REQUEST_DELAY = 1   # detik
# Max thread paralel untuk scoring
_MAX_WORKERS   = 4


# ------------------------------------------------------------------
# Step 1: Ambil top N symbol by 24h quote volume
# ------------------------------------------------------------------

def get_top_symbols(top_n: int = SCAN_TOP_N) -> list[str]:
    """
    Return list symbol USDT perpetual futures sorted by 24h quoteVolume,
    ambil top_n terbesar.
    """
    url  = f"{BINANCE_BASE_URL}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    tickers = resp.json()

    # Filter: hanya USDT perp, exclude stablecoin pair
    exclude_suffix = {"BUSDUSDT", "USDCUSDT", "TUSDUSDT", "FDUSDUSDT"}
    usdt_perps = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in exclude_suffix
    ]

    # Sort by quoteVolume descending
    usdt_perps.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)

    symbols = [t["symbol"] for t in usdt_perps[:top_n]]
    logger.info("[scanner] Top %d symbols fetched: %s...", top_n, symbols[:5])
    return symbols


# ------------------------------------------------------------------
# Step 2: Scoring satu token
# ------------------------------------------------------------------

def score_symbol(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    kline_limit: int = 210,
) -> Optional[dict]:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = get_df(symbol, interval=interval, limit=kline_limit)
            if df is None or len(df) < 50:
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

            time.sleep(_REQUEST_DELAY)
            from indicators.funding import fetch_funding_rate, score_funding, get_funding_detail
            fund_df      = fetch_funding_rate(symbol, limit=90)
            scores["funding"] = float(score_funding(fund_df))
            funding_detail    = get_funding_detail(fund_df)

            time.sleep(_REQUEST_DELAY)
            from indicators.lsr import fetch_lsr, score_lsr, get_lsr_detail
            lsr_df       = fetch_lsr(symbol, interval=interval, limit=96)
            scores["lsr"]  = float(score_lsr(lsr_df))
            lsr_detail     = get_lsr_detail(lsr_df)

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

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 2  # 2s, 4s
                logger.warning("[scanner] %s attempt %d failed: %s — retry in %ds", 
                               symbol, attempt + 1, e, wait)
                time.sleep(wait)
            else:
                logger.warning("[scanner] %s scoring failed after %d attempts: %s", 
                               symbol, max_retries, e)
                return None


# ------------------------------------------------------------------
# Step 3: Scan semua top symbols, parallel
# ------------------------------------------------------------------

def scan(
    top_n:     int   = SCAN_TOP_N,
    interval:  str   = DEFAULT_INTERVAL,
    threshold: float = SCAN_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Scan top_n token, return daftar yang lolos threshold,
    sorted by abs(weighted_total) descending.

    Returns:
        list[dict] — tiap dict = hasil score_symbol yang lolos
    """
    symbols = get_top_symbols(top_n)
    logger.info("[scanner] Scanning %d symbols (interval=%s, threshold=%.1f)...",
                len(symbols), interval, threshold)

    results   = []
    failed    = 0

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
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

    # Filter threshold
    passed = [
        r for r in results
        if abs(r["weighted_total"]) >= threshold
        and r["direction"] != "NEUTRAL"
    ]

    # Sort by abs score descending
    passed.sort(key=lambda x: abs(x["weighted_total"]), reverse=True)

    logger.info(
        "[scanner] Done. %d/%d symbols scored, %d passed threshold, %d failed.",
        len(results), len(symbols), len(passed), failed,
    )
    return passed


# ------------------------------------------------------------------
# Format ringkasan scan untuk Telegram
# ------------------------------------------------------------------

def format_scan_summary(passed: list[dict], top_n: int, interval: str) -> str:
    """Format hasil scan jadi satu pesan Telegram HTML."""
    if not passed:
        return (
            f"🔍 <b>Scan selesai</b> — {top_n} token ({interval})\n"
            f"⚪ Tidak ada token yang lolos threshold."
        )

    lines = [
        f"🔍 <b>Scan Result</b> — Top {top_n} ({interval})",
        f"✅ <b>{len(passed)} token lolos</b> → masuk pipeline\n",
    ]
    for i, r in enumerate(passed[:20], 1):   # tampilkan max 20
        sym   = r["symbol"]
        total = r["weighted_total"]
        dir_  = r["direction"]
        emoji = "🟢" if dir_ == "LONG" else "🔴"
        lines.append(f"  {i:>2}. {emoji} <b>{sym:<14}</b> score: <code>{total:+.2f}</code>")

    if len(passed) > 20:
        lines.append(f"  ... dan {len(passed) - 20} lainnya")

    return "\n".join(lines)
