"""
ai/analyst.py — Kirim konteks ke DeepSeek, parse keputusan trading.
"""

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


def _build_prompt(pred: dict, bt_result: dict, train_result: dict) -> str:
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
        f"  {k:<8}: score={v:+.0f}, weight={weights.get(k, 1.0):.4f}, "
        f"contrib={v * weights.get(k, 1.0):+.4f}"
        for k, v in scores.items()
    )

    prompt = f"""You are an expert crypto futures trader and quantitative analyst.
Analyze the data below and decide whether to LONG, SHORT, or HOLD on {symbol} futures.
Output ONLY a valid JSON object. No markdown, no explanation outside the JSON.

=== SYMBOL & TIMEFRAME ===
Symbol   : {symbol}
Interval : {interval}

=== CURRENT MARKET STATE ===
Current price  : {cur_price}
ML prediction  : {direction} with {conf:.1f}% confidence
Predicted price (3 candles ahead): {pred_price}
Weighted total score: {w_total:+.4f}

=== INDICATOR SCORES (ML-adjusted weights) ===
{score_lines}

=== BACKTEST RESULTS ===
{bt_sum}

=== OHLCV + TECHNICAL DATA (last {MAX_CANDLES_FOR_AI} candles) ===
{candle_csv}

=== RULES ===
- Only enter if edge is clear.
- leverage must be integer between 1 and {DEFAULT_LEVERAGE}.
- reason must be minimum 3 sentences in English.

Return EXACTLY this JSON (no markdown):
{{
  "action": "BUYING" or "SELLING" or "HOLD",
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "leverage": <int>,
  "reason": "<detailed explanation>"
}}"""

    return prompt


def _call_deepseek(prompt: str) -> str:
    """Call DeepSeek API. Timeout 300s karena deepseek-reasoner bisa lambat."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }

    # deepseek-reasoner tidak support temperature parameter
    payload = {
        "model":      DEEPSEEK_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    # Kalau bukan reasoner, tambahkan temperature
    if "reasoner" not in DEEPSEEK_MODEL:
        payload["temperature"] = 0.0

    url  = f"{DEEPSEEK_BASE_URL}/chat/completions"
    logger.info("[analyst] Sending request to DeepSeek (model=%s, timeout=300s)...", DEEPSEEK_MODEL)

    resp = requests.post(url, headers=headers, json=payload, timeout=300)

    logger.debug("[analyst] DeepSeek HTTP %d | Body[:300]: %s",
                 resp.status_code, resp.text[:300])

    if resp.status_code != 200:
        raise requests.HTTPError(
            f"DeepSeek HTTP {resp.status_code}: {resp.text[:300]}",
            response=resp,
        )

    data = resp.json()

    # deepseek-reasoner punya reasoning_content + content
    # ambil content (bukan reasoning)
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"DeepSeek response tidak punya choices: {data}")

    msg = choices[0].get("message", {})

    # Coba content dulu, fallback ke reasoning_content
    content = msg.get("content", "") or msg.get("reasoning_content", "")

    if not content or not content.strip():
        logger.error("[analyst] DeepSeek content kosong. Full response: %s", json.dumps(data)[:500])
        raise ValueError(f"DeepSeek content kosong. Response: {json.dumps(data)[:300]}")

    logger.info("[analyst] DeepSeek content[:200]: %s", content[:200])
    return content


def _parse_ai_response(raw: str) -> dict:
    # 1. Direct parse
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


def _validate(parsed: dict) -> dict:
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


def analyze(pred: dict, bt_result: dict, train_result: dict) -> dict:
    symbol = pred["symbol"]
    logger.info("[analyst] Calling DeepSeek for %s...", symbol)

    try:
        prompt    = _build_prompt(pred, bt_result, train_result)
        raw       = _call_deepseek(prompt)
        parsed    = _parse_ai_response(raw)
        validated = _validate(parsed)

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
