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

def _btc_context_bonus(abs_weighted_total: float) -> float:

    if abs_weighted_total < 1.0:
        return 0.0
    elif abs_weighted_total < 3.0:
        return 0.5
    elif abs_weighted_total < 6.0:
        return 1.0
    elif abs_weighted_total < 8.0:
        return 1.5
    else:
        return 2.0


def get_btc_context(interval: str = DEFAULT_INTERVAL) -> tuple[float, float]:

    logger.info("[scanner] Scanning BTCUSDT untuk market context...")
    result = score_symbol("BTCUSDT", interval=interval)

    if result is None:
        logger.warning("[scanner] BTC scan gagal, bonus default 0.0")
        return 0.0, 0.0

    btc_total = result["weighted_total"]
    bonus     = _btc_context_bonus(abs(btc_total))

    signed_bonus = bonus if btc_total >= 0 else -bonus

    logger.info(
        "[scanner] BTC weighted_total=%.4f → additive_bonus=%+.2f",
        btc_total, signed_bonus,
    )
    return btc_total, signed_bonus


def apply_btc_context(weighted_total: float, btc_bonus: float) -> float:

    if btc_bonus == 0.0:
        return weighted_total

    if (btc_bonus > 0 and weighted_total > 0) or \
       (btc_bonus < 0 and weighted_total < 0):
        return round(weighted_total + btc_bonus, 4)

    return round(weighted_total - abs(btc_bonus), 4)

_MIN_CANDLES_FOR_PIPELINE = 100

_PUMP_DUMP_CANDLES   = 12
_PUMP_DUMP_THRESHOLD = 0.30 


def _is_pump_or_dump(df) -> tuple[bool, float]:

    if len(df) < _PUMP_DUMP_CANDLES:
        return False, 0.0

    window = df.iloc[-_PUMP_DUMP_CANDLES:]

    close_start = float(window["close"].iloc[0])
    if close_start <= 0:
        return False, 0.0

    highest_high = float(window["high"].max())
    lowest_low   = float(window["low"].min())

    pump_pct = (highest_high - close_start) / close_start   # selalu >= 0
    dump_pct = (lowest_low  - close_start) / close_start    # selalu <= 0

    if pump_pct >= _PUMP_DUMP_THRESHOLD:
        return True, pump_pct   # PUMP — high melewati threshold

    if abs(dump_pct) >= _PUMP_DUMP_THRESHOLD:
        return True, dump_pct   # DUMP — low melewati threshold

    return False, 0.0

def _get_listed_symbols() -> set[str]:

    import time as _time
    try:
        url  = f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        now_ms    = int(_time.time() * 1000)
        min_age   = 30 * 24 * 60 * 60 * 1000  # 30 hari dalam ms
        old_enough = set()
        for s in resp.json().get("symbols", []):
            onboard = s.get("onboardDate", 0)
            if onboard and (now_ms - onboard) >= min_age:
                old_enough.add(s["symbol"])
        logger.info("[scanner] exchangeInfo: %d simbol sudah listed >= 30 hari", len(old_enough))
        return old_enough
    except Exception as e:
        logger.warning("[scanner] Gagal fetch exchangeInfo untuk age filter: %s — skip filter", e)
        return set() 


def get_top_symbols(top_n: int = SCAN_TOP_N) -> list[str]:
    old_enough = _get_listed_symbols()

    url  = f"{BINANCE_BASE_URL}/fapi/v1/ticker/24hr"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    tickers = resp.json()

    usdt_perps = [
        t for t in tickers
        if t["symbol"].endswith("USDT")
        and t["symbol"] not in SCANNER_BLACKLIST
        and (not old_enough or t["symbol"] in old_enough)
    ]
    usdt_perps.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
    symbols = [t["symbol"] for t in usdt_perps[:top_n]]
    logger.info(
        "[scanner] Top %d symbols (blacklist + age filter excluded): %s...",
        top_n, symbols[:5]
    )
    return symbols

def score_symbol(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    kline_limit: int = 210,
) -> Optional[dict]:
    max_retries = 3

    for attempt in range(max_retries):
        try:
            df = get_df(symbol, interval=interval, limit=kline_limit)

            if df is None or len(df) < _MIN_CANDLES_FOR_PIPELINE:
                logger.info(
                    "[scanner] %s: skip — hanya %d candle tersedia (butuh >= %d). ",
                    symbol,
                    len(df) if df is not None else 0,
                    _MIN_CANDLES_FOR_PIPELINE,
                )
                return None

            pumped, pct = _is_pump_or_dump(df)
            if pumped:
                tag = "PUMP 🚀" if pct > 0 else "DUMP 💀"
                logger.info(
                    "[scanner] %s: skip — %s %.1f%% dalam %d candle terakhir.",
                    symbol, tag, abs(pct) * 100, _PUMP_DUMP_CANDLES,
                )
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

            time.sleep(SCANNER_REQUEST_DELAY)
            fund_df = fetch_funding_rate(symbol, limit=90)
            scores["funding"] = float(score_funding(fund_df))
            funding_detail    = get_funding_detail(fund_df)

            time.sleep(SCANNER_REQUEST_DELAY)
            lsr_df = fetch_lsr(symbol, interval=interval, limit=96)
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

        except requests.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status in (429, 418):
                wait = 10 * (attempt + 1)
                logger.warning("[scanner] %s rate limited (HTTP %d) — tunggu %ds", symbol, status, wait)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logger.warning("[scanner] %s gagal setelah %d attempt: %s", symbol, max_retries, e)
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logger.warning("[scanner] %s scoring failed: %s", symbol, e)
                return None

    return None

def scan(
    top_n:     int   = SCAN_TOP_N,
    interval:  str   = DEFAULT_INTERVAL,
    threshold: float = SCAN_SCORE_THRESHOLD,
) -> list[dict]:
    btc_total, btc_bonus = get_btc_context(interval)

    symbols = get_top_symbols(top_n)
    logger.info(
        "[scanner] Scanning %d symbols (interval=%s, threshold=%.1f, btc_bonus=%+.2f)...",
        len(symbols), interval, threshold, btc_bonus,
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
    for r in results:
        original = r["weighted_total"]
        adjusted = apply_btc_context(original, btc_bonus)
        r["weighted_total_raw"]     = original
        r["weighted_total"]         = adjusted
        r["btc_bonus_applied"]      = btc_bonus
        r["direction"] = (
            "LONG"    if adjusted > 0 else
            "SHORT"   if adjusted < 0 else
            "NEUTRAL"
        )

    passed = [
        r for r in results
        if abs(r["weighted_total"]) >= threshold
        and r["direction"] != "NEUTRAL"
    ]
    passed.sort(key=lambda x: abs(x["weighted_total"]), reverse=True)

    logger.info(
        "[scanner] Done. BTC=%.4f (bonus=%+.2f) | %d/%d scored | %d lolos | %d gagal.",
        btc_total, btc_bonus, len(results), len(symbols), len(passed), failed,
    )
    return passed

def format_scan_summary(passed: list[dict], top_n: int, interval: str) -> str:
    if not passed:
        return (
            f"🔍 <b>Scan selesai</b> — {top_n} token ({interval})\n"
            f"⚪ Tidak ada token yang lolos threshold."
        )

    btc_bonus = passed[0].get("btc_bonus_applied", 0.0) if passed else 0.0
    btc_label = (
        f"🟢 Bullish (+{abs(btc_bonus):.2f})" if btc_bonus > 0 else
        f"🔴 Bearish (-{abs(btc_bonus):.2f})" if btc_bonus < 0 else
        "⚪ Netral (no adjustment)"
    )

    lines = [
        f"🔍 <b>Scan Result</b> — Top {top_n} ({interval})",
        f"₿  BTC Context : {btc_label} <i>(additive)</i>",
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
