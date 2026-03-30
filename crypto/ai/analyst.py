"""
ai/analyst.py — AI hanya memutuskan arah (BUYING/SELLING/SKIP) + reason.
SL, TP, leverage, qty_fraction semuanya dari Kelly+MC+ATR.
"""

import json
import logging
import os
import re
import sys
import time

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    RISK_PER_TRADE_PCT,
)
from ml.kelly import compute_position, format_for_prompt

logger = logging.getLogger(__name__)

MAX_CANDLES_FOR_AI = 200
MAX_RETRIES        = 2
RETRY_DELAY        = 3

# Buffer di atas breakeven winrate — cukup 3% agar tidak terlalu ketat
WINRATE_BUFFER     = 0.03

SYSTEM_PROMPT = """\
You are a crypto futures trading execution engine.
Your ONLY job is to output a single JSON object based on the data provided.
You MUST NOT output any text, explanation, reasoning, or commentary outside the JSON.
You MUST NOT use markdown, code fences, or any wrapper around the JSON.
Your first character of output MUST be '{' and your last character MUST be '}'."""


# ------------------------------------------------------------------
# Dynamic winrate threshold
# ------------------------------------------------------------------

def _min_winrate(rr: float, buffer: float = WINRATE_BUFFER) -> float:
    """
    Hitung minimum winrate yang profitable berdasarkan RR aktual.

    Breakeven formula: winrate_min = 1 / (1 + RR)
    Dengan buffer keamanan 3%:
      RR 2.0 → breakeven 33.3% + 3% = 36.3%
      RR 1.5 → breakeven 40.0% + 3% = 43.0%
      RR 1.0 → breakeven 50.0% + 3% = 53.0%

    Floor 30% — tidak boleh lebih rendah dari ini.
    Ceil  55% — tidak boleh lebih tinggi dari ini (jangan terlalu ketat).
    """
    breakeven = 1.0 / (1.0 + rr)
    threshold = breakeven + buffer
    return round(max(0.30, min(threshold, 0.55)), 4)


# ------------------------------------------------------------------
# Build candle CSV
# ------------------------------------------------------------------

def _build_candle_csv(context_df) -> str:
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "transactions",
        "ma10", "ma20", "ma50", "rsi14",
        "vol_ma10", "vol_ma20", "freq_ma10", "freq_ma20",
    ]
    available = [c for c in cols if c in context_df.columns]
    df_tail   = context_df[available].tail(MAX_CANDLES_FOR_AI).copy()
    round_map = {
        "open": 4, "high": 4, "low": 4, "close": 4,
        "volume": 2, "transactions": 0,
        "ma10": 4, "ma20": 4, "ma50": 4, "rsi14": 2,
        "vol_ma10": 2, "vol_ma20": 2,
        "freq_ma10": 1, "freq_ma20": 1,
    }
    for col, dec in round_map.items():
        if col in df_tail.columns:
            df_tail[col] = df_tail[col].round(dec)
    return df_tail.to_csv(index=False)


# ------------------------------------------------------------------
# Build prompt
# ------------------------------------------------------------------

def _build_prompt(
    pred: dict,
    bt_result: dict,
    pos_long: dict,
    pos_short: dict,
) -> str:
    symbol     = pred["symbol"]
    interval   = pred["interval"]
    direction  = pred["direction"]
    conf       = pred["confidence"] * 100
    cur_price  = pred["current_price"]
    pred_price = pred["predicted_price"]
    scores     = pred["scores"]
    weights    = pred["weights"]
    w_total    = pred["weighted_total"]
    bt_sum     = bt_result["summary_text"]
    candle_csv = _build_candle_csv(pred["context_df"])

    score_lines = "\n".join(
        f"  {k:<8}: score={v:+.0f}  weight={weights.get(k,1.0):.4f}  contrib={v*weights.get(k,1.0):+.4f}"
        for k, v in scores.items()
    )

    # Dynamic threshold per arah berdasarkan RR masing-masing
    rr_long       = pos_long["rr_ratio"]
    rr_short      = pos_short["rr_ratio"]
    min_wr_long   = _min_winrate(rr_long)
    min_wr_short  = _min_winrate(rr_short)

    # Winrate aktual dari backtest
    bt_after      = bt_result["after"]
    wr_long_pct   = bt_after.get("winrate_up", 0) * 100
    wr_short_pct  = bt_after.get("winrate_dn", 0) * 100

    return f"""=== SYMBOL ===
{symbol} | {interval}

=== MARKET STATE ===
Current price : {cur_price}
ML prediction : {direction} | confidence={conf:.1f}%
Predicted price (3 candles ahead): {pred_price}
Weighted score: {w_total:+.4f}

=== INDICATOR SCORES ===
{score_lines}

=== BACKTEST ===
{bt_sum}

=== IF BUYING (pre-calculated by system) ===
{format_for_prompt(pos_long)}

=== IF SELLING (pre-calculated by system) ===
{format_for_prompt(pos_short)}

=== WINRATE THRESHOLDS (dynamic, based on RR) ===
  BUYING  → RR={rr_long:.2f}  | min winrate required={min_wr_long*100:.1f}% | actual={wr_long_pct:.1f}%  | {'✓ PASS' if wr_long_pct/100 >= min_wr_long else '✗ FAIL'}
  SELLING → RR={rr_short:.2f} | min winrate required={min_wr_short*100:.1f}% | actual={wr_short_pct:.1f}% | {'✓ PASS' if wr_short_pct/100 >= min_wr_short else '✗ FAIL'}

=== OHLCV (last {MAX_CANDLES_FOR_AI} candles, {interval}) ===
{candle_csv}

=== YOUR JOB ===
Decide BUYING, SELLING, or SKIP based on all data above.
SL, TP, leverage, qty_fraction are already calculated — do NOT change them.
Output EXACTLY this JSON and nothing else:
{{
  "action": "BUYING" or "SELLING" or "SKIP",
  "reason": "<2 sentences max, facts only: prices, winrates, edge, indicator values. No opinions.>"
}}

SKIP only if ANY of these conditions are true:
  - Edge is negative for chosen direction
  - Actual winrate is below the minimum required winrate shown above
  - ML confidence < 55%
  - No clear signal from indicators"""


# ------------------------------------------------------------------
# Call DeepSeek
# ------------------------------------------------------------------

def _call_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       DEEPSEEK_MODEL,
        "temperature": 0.0,
        "max_tokens":  256,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }

    url        = f"{DEEPSEEK_BASE_URL}/chat/completions"
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("[analyst] DeepSeek attempt %d/%d...", attempt, MAX_RETRIES)
        resp = requests.post(url, headers=headers, json=payload, timeout=120)

        if resp.status_code != 200:
            raise requests.HTTPError(
                f"DeepSeek HTTP {resp.status_code}: {resp.text[:300]}", response=resp)

        data    = resp.json()
        choices = data.get("choices", [])
        if not choices:
            last_error = ValueError(f"Tidak ada choices: {json.dumps(data)[:200]}")
            logger.warning("[analyst] attempt %d: %s", attempt, last_error)
            time.sleep(RETRY_DELAY)
            continue

        content = (choices[0].get("message", {}).get("content") or "").strip()
        if not content:
            last_error = ValueError("Content kosong")
            logger.warning("[analyst] attempt %d: content kosong", attempt)
            time.sleep(RETRY_DELAY)
            continue

        logger.info("[analyst] OK (%d chars): %s...", len(content), content[:80])
        return content

    raise last_error or ValueError("Semua retry gagal")


# ------------------------------------------------------------------
# Parse & validate
# ------------------------------------------------------------------

def _parse(raw: str) -> dict:
    for candidate in [
        raw.strip(),
        re.sub(r"```(?:json)?|```", "", raw, flags=re.IGNORECASE).strip(),
    ]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.error("[analyst] Parse gagal. Raw:\n%s", raw)
    raise ValueError(f"Tidak bisa parse JSON (len={len(raw)})")


def _validate_action(parsed: dict) -> str:
    action = str(parsed.get("action", "SKIP")).upper()
    if action in ("HOLD", "NONE", "NEUTRAL", "NO_TRADE"):
        action = "SKIP"
    if action not in ("BUYING", "SELLING", "SKIP"):
        logger.warning("[analyst] action tidak dikenal (%s) → SKIP", action)
        action = "SKIP"
    return action


# ------------------------------------------------------------------
# Public: analyze
# ------------------------------------------------------------------

def analyze(pred: dict, bt_result: dict, train_result: dict) -> dict:
    """
    Kirim data ke AI — AI return action + reason saja.
    SL/TP/leverage/qty_fraction dihitung dari Kelly+MC+ATR.
    Winrate threshold dinamis berdasarkan RR aktual masing-masing arah.
    """
    symbol   = pred["symbol"]
    raw_df   = train_result["raw_df"]
    bt_after = bt_result["after"]
    risk_max = RISK_PER_TRADE_PCT / 100

    logger.info("[analyst] Computing position sizing for %s...", symbol)

    # Pre-calculate posisi untuk kedua arah
    pos_long = compute_position(
        df=raw_df,
        direction="LONG",
        winrate=float(bt_after.get("winrate_up", 0.5)),
        risk_per_trade=risk_max,
        max_fraction=risk_max,
    )
    pos_short = compute_position(
        df=raw_df,
        direction="SHORT",
        winrate=float(bt_after.get("winrate_dn", 0.5)),
        risk_per_trade=risk_max,
        max_fraction=risk_max,
    )

    # Log threshold yang akan dipakai
    min_wr_long  = _min_winrate(pos_long["rr_ratio"])
    min_wr_short = _min_winrate(pos_short["rr_ratio"])
    logger.info(
        "[analyst] %s dynamic winrate threshold — LONG: %.1f%% (RR=%.2f) | SHORT: %.1f%% (RR=%.2f)",
        symbol,
        min_wr_long * 100,  pos_long["rr_ratio"],
        min_wr_short * 100, pos_short["rr_ratio"],
    )

    logger.info("[analyst] Calling DeepSeek for %s...", symbol)

    try:
        prompt = _build_prompt(pred, bt_result, pos_long, pos_short)
        raw    = _call_deepseek(prompt)
        parsed = _parse(raw)
        action = _validate_action(parsed)
        reason = str(parsed.get("reason", ""))

        # Pilih posisi sesuai keputusan AI
        if action == "BUYING":
            pos = pos_long
        elif action == "SELLING":
            pos = pos_short
        else:
            pos = pos_long  # SKIP — dummy, tidak dieksekusi

        return {
            "ok":             True,
            "raw_response":   raw,
            "action":         action,
            "entry_price":    pos["entry_price"],
            "stop_loss":      pos["stop_loss"],
            "take_profit":    pos["take_profit"],
            "leverage":       pos["leverage"],
            "qty_fraction":   pos["qty_fraction"],
            "reason":         reason,
            "position_detail": pos,
        }

    except requests.HTTPError as e:
        msg = f"DeepSeek HTTP error: {e}"
        logger.error("[analyst] %s", msg)
        return {"ok": False, "reason_fail": msg, "action": "SKIP"}

    except ValueError as e:
        msg = str(e)
        logger.error("[analyst] %s", msg)
        return {"ok": False, "reason_fail": msg, "action": "SKIP"}

    except Exception as e:
        msg = f"Unexpected error: {e}"
        logger.exception("[analyst] %s", msg)
        return {"ok": False, "reason_fail": msg, "action": "SKIP"}
