
import json
import logging
import os
import re
import sys

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEFAULT_LEVERAGE,
)

logger = logging.getLogger(__name__)

MAX_CANDLES_FOR_AI = 200 


# ------------------------------------------------------------------
# Build prompt
# ------------------------------------------------------------------

def _build_candle_csv(context_df) -> str:
    """Konversi DataFrame ke CSV ringkas untuk prompt."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "transactions",
        "ma10", "ma20", "ma50", "rsi14",
        "vol_ma10", "vol_ma20", "freq_ma10", "freq_ma20",
    ]
    # Pakai kolom yang ada
    available = [c for c in cols if c in context_df.columns]
    df_tail   = context_df[available].tail(MAX_CANDLES_FOR_AI).copy()

    # Round untuk hemat token
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


def _build_prompt(pred: dict, bt_result: dict, train_result: dict) -> str:
    symbol    = pred["symbol"]
    interval  = pred["interval"]
    direction = pred["direction"]
    conf      = pred["confidence"] * 100
    cur_price = pred["current_price"]
    pred_price = pred["predicted_price"]
    scores    = pred["scores"]
    weights   = pred["weights"]
    w_total   = pred["weighted_total"]
    bt_after  = bt_result["after"]
    bt_sum    = bt_result["summary_text"]
    candle_csv = _build_candle_csv(pred["context_df"])

    # Skor indikator formatted
    score_lines = "\n".join(
        f"  {k:<6}: score={v:+.0f}, weight={weights.get(k, 1.0):.4f}, "
        f"weighted_contrib={v * weights.get(k, 1.0):+.4f}"
        for k, v in scores.items()
    )

    prompt = f"""You are an expert crypto futures trader and quantitative analyst.
Your task: analyze the data below and decide whether to LONG, SHORT, or HOLD on {symbol} futures.
Output ONLY a valid JSON object. No explanation outside the JSON.

=== SYMBOL & TIMEFRAME ===
Symbol   : {symbol}
Interval : {interval}
Candles  : last {MAX_CANDLES_FOR_AI} provided below

=== CURRENT MARKET STATE ===
Current price  : {cur_price}
ML prediction  : {direction} with {conf:.1f}% confidence
Predicted price (3 candles ahead): {pred_price}
Weighted total score: {w_total:+.4f}

=== INDICATOR SCORES (ML-adjusted weights) ===
{score_lines}

=== BACKTEST RESULTS (ML-adjusted weights, {train_result['n_candles']} candles) ===
{bt_sum}

=== OHLCV + TECHNICAL DATA (last {MAX_CANDLES_FOR_AI} candles, {interval}) ===
Columns: open_time(ms), open, high, low, close, volume, transactions,
         ma10, ma20, ma50, rsi14, vol_ma10, vol_ma20, freq_ma10, freq_ma20

{candle_csv}

=== DECISION RULES ===
- Analyze all data holistically: price action, indicators, volume, ML prediction.
- Use the backtest winrates to calibrate your confidence.
- Be conservative: only enter if edge is clear.
- Set realistic stop_loss and take_profit based on recent volatility (ATR-like logic).
- leverage must be between 1 and {DEFAULT_LEVERAGE}.
- reason must be detailed (minimum 3 sentences): explain WHY you chose the action,
  what signals confirmed it, and what risks you see.

=== REQUIRED OUTPUT FORMAT ===
Return EXACTLY this JSON (no markdown, no extra text):
{{
  "action": "BUYING" or "SELLING" or "HOLD",
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "leverage": <int 1-{DEFAULT_LEVERAGE}>,
  "reason": "<detailed explanation in English, minimum 3 sentences>"
}}"""

    return prompt


# ------------------------------------------------------------------
# Call DeepSeek R1
# ------------------------------------------------------------------

def _call_deepseek(prompt: str) -> str:
    """Call DeepSeek API, return raw content string."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       DEEPSEEK_MODEL,
        "temperature": 0.0,
        "max_tokens":  2048,
        "messages": [
            {
                "role":    "user",
                "content": prompt,
            }
        ],
    }
    url  = f"{DEEPSEEK_BASE_URL}/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ------------------------------------------------------------------
# Parse JSON dari output AI
# ------------------------------------------------------------------

def _parse_ai_response(raw: str) -> dict:
    """
    Parse JSON dari response AI.
    Coba beberapa strategi: direct parse, extract dari markdown fence,
    extract pola {...}.
    """
    # 1. Direct
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fence
    fence = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(fence)
    except json.JSONDecodeError:
        pass

    # 3. Cari blok {...}
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Tidak bisa parse JSON dari AI response:\n{raw[:500]}")


# ------------------------------------------------------------------
# Validasi output AI
# ------------------------------------------------------------------

def _validate(parsed: dict) -> dict:
    """Validasi dan normalize field output AI."""
    action = str(parsed.get("action", "HOLD")).upper()
    if action not in ("BUYING", "SELLING", "HOLD"):
        action = "HOLD"

    return {
        "action":       action,
        "entry_price":  float(parsed.get("entry_price", 0)),
        "stop_loss":    float(parsed.get("stop_loss",   0)),
        "take_profit":  float(parsed.get("take_profit", 0)),
        "leverage":     max(1, min(int(parsed.get("leverage", DEFAULT_LEVERAGE)), DEFAULT_LEVERAGE)),
        "reason":       str(parsed.get("reason", "")),
    }


# ------------------------------------------------------------------
# Public: analyze
# ------------------------------------------------------------------

def analyze(pred: dict, bt_result: dict, train_result: dict) -> dict:
    """
    Kirim data ke DeepSeek R1, parse keputusan.

    Returns:
        {
          "ok": bool,
          "reason_fail": str (jika ok=False),
          "action":       "BUYING" | "SELLING" | "HOLD",
          "entry_price":  float,
          "stop_loss":    float,
          "take_profit":  float,
          "leverage":     int,
          "reason":       str,   ← alasan detail dari AI
          "raw_response": str,   ← response mentah dari AI
        }
    """
    symbol = pred["symbol"]
    logger.info("[analyst] Calling DeepSeek R1 for %s...", symbol)

    try:
        prompt      = _build_prompt(pred, bt_result, train_result)
        raw         = _call_deepseek(prompt)
        logger.info("[analyst] Raw response: %s", raw[:300])
        parsed      = _parse_ai_response(raw)
        validated   = _validate(parsed)

        return {
            "ok":           True,
            "raw_response": raw,
            **validated,
        }

    except requests.HTTPError as e:
        msg = f"DeepSeek HTTP error: {e}"
        logger.error("[analyst] %s", msg)
        return {"ok": False, "reason_fail": msg, "action": "HOLD"}

    except ValueError as e:
        msg = str(e)
        logger.error("[analyst] %s", msg)
        return {"ok": False, "reason_fail": msg, "action": "HOLD"}

    except Exception as e:
        msg = f"Unexpected error: {e}"
        logger.exception("[analyst] %s", msg)
        return {"ok": False, "reason_fail": msg, "action": "HOLD"}
