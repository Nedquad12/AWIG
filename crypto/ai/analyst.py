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
from ml.kelly           import compute_position, format_for_prompt, RISK_PER_TRADE_PCT as KELLY_RISK
from ml.regime_detector import format_for_ai as regime_format_for_ai
from ml.wfv             import TRAIN_END_IDX

logger = logging.getLogger(__name__)

MAX_CANDLES_FOR_AI = 100
MAX_RETRIES        = 2
RETRY_DELAY        = 3

SYSTEM_PROMPT = """\
You are a crypto futures trading execution engine.
Your ONLY job is to output a single JSON object based on the data provided.
You MUST NOT output any text, explanation, reasoning, or commentary outside the JSON.
You MUST NOT use markdown, code fences, or any wrapper around the JSON.
Your first character of output MUST be '{' and your last character MUST be '}'."""


def _min_winrate(rr: float) -> float:
    breakeven = 1.0 / (1.0 + rr)
    return round(max(0.20, min(breakeven, 0.55)), 4)


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
        "vol_ma10": 2, "vol_ma20": 2, "freq_ma10": 1, "freq_ma20": 1,
    }
    for col, dec in round_map.items():
        if col in df_tail.columns:
            df_tail[col] = df_tail[col].round(dec)
    return df_tail.to_csv(index=False)


def _build_prompt(pred, wfv_result, train_result, pos_long, pos_short) -> str:
    symbol     = pred["symbol"]
    interval   = pred["interval"]
    direction  = pred["direction"]
    conf       = pred["confidence"] * 100
    cur_price  = pred["current_price"]
    pred_price = pred["predicted_price"]
    scores     = pred["scores"]
    weights    = pred["weights"]
    w_total    = pred["weighted_total"]
    candle_csv = _build_candle_csv(pred["context_df"])

    score_lines = "\n".join(
        f"  {k:<8}: score={v:+.0f}  weight={weights.get(k,1.0):.4f}  contrib={v*weights.get(k,1.0):+.4f}"
        for k, v in scores.items()
    )

    rr_long      = pos_long["rr_ratio"]
    rr_short     = pos_short["rr_ratio"]
    min_wr_long  = _min_winrate(rr_long)
    min_wr_short = _min_winrate(rr_short)

    wfv_after    = wfv_result["after"]
    wr_long_pct  = wfv_after.get("winrate_up", 0) * 100
    wr_short_pct = wfv_after.get("winrate_dn", 0) * 100

    wr_long_note  = f" ⚠️ RAW={pos_long['winrate_raw']*100:.1f}%"  if pos_long.get("winrate_warning")  else ""
    wr_short_note = f" ⚠️ RAW={pos_short['winrate_raw']*100:.1f}%" if pos_short.get("winrate_warning") else ""

    # Regime + WFV context untuk AI
    regime_info  = train_result.get("regime_info", {})
    wfv_folds    = wfv_result.get("ok_folds", [])
    regime_block = regime_format_for_ai(regime_info, wfv_folds) if regime_info else "Regime: N/A"

    # WFV summary
    wfv_pnl      = wfv_result.get("total_net_pnl", 0)
    wfv_wr       = wfv_result.get("overall_wr", 0) * 100
    wfv_folds_n  = wfv_result.get("n_folds", 0)
    wfv_trades   = wfv_result.get("total_trades", 0)

    return f"""=== SYMBOL ===
{symbol} | {interval}

=== MARKET STATE ===
Current price : {cur_price}
ML prediction : {direction} | confidence={conf:.1f}%
Predicted price (3 candles ahead): {pred_price}
Weighted score: {w_total:+.4f}

=== INDICATOR SCORES (9 indicators) ===
{score_lines}
NOTE: funding & lsr are real-time sentiment filters, not ML features.

{regime_block}

=== WALK-FORWARD VALIDATION SUMMARY ===
Config  : {wfv_folds_n} folds | train=500 | test=100 | step=100
Overall : WR={wfv_wr:.1f}% | PnL_net=${wfv_pnl:+.2f} | {wfv_trades} trades (modal $100/trade)
Fee     : 0.1% each side (0.2% RT) | Slippage: live orderbook

=== IF BUYING (pre-calculated by system) ===
{format_for_prompt(pos_long)}

=== IF SELLING (pre-calculated by system) ===
{format_for_prompt(pos_short)}

=== WINRATE THRESHOLDS (dynamic, based on RR) ===
  BUYING  → RR={rr_long:.2f}  | min winrate={min_wr_long*100:.1f}% | actual={wr_long_pct:.1f}%{wr_long_note}  | {'✓ PASS' if wr_long_pct/100 >= min_wr_long else '✗ FAIL'}
  SELLING → RR={rr_short:.2f} | min winrate={min_wr_short*100:.1f}% | actual={wr_short_pct:.1f}%{wr_short_note} | {'✓ PASS' if wr_short_pct/100 >= min_wr_short else '✗ FAIL'}

=== OHLCV (last {MAX_CANDLES_FOR_AI} candles, {interval}) ===
{candle_csv}

=== YOUR JOB ===
Decide BUYING, SELLING, or SKIP based on ALL data above.
Pay special attention to:
  1. WFV history — if current regime is historically unprofitable, SKIP.
  2. Winrate thresholds — if FAIL, lean toward SKIP.
  3. Edge — if negative, SKIP.

SL, TP, leverage, qty_fraction are already calculated — do NOT change them.
Output EXACTLY this JSON and nothing else:
{{
  "action": "BUYING" or "SELLING" or "SKIP",
  "reason": "<2 sentences max, cite specific numbers: WFV PnL, regime, winrate, edge>"
}}"""


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
            time.sleep(RETRY_DELAY)
            continue
        content = (choices[0].get("message", {}).get("content") or "").strip()
        if not content:
            last_error = ValueError("Content kosong")
            time.sleep(RETRY_DELAY)
            continue
        logger.info("[analyst] OK (%d chars): %s...", len(content), content[:80])
        return content

    raise last_error or ValueError("Semua retry gagal")


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
    raise ValueError(f"Tidak bisa parse JSON (len={len(raw)})")


def _validate_action(parsed: dict) -> str:
    action = str(parsed.get("action", "SKIP")).upper()
    if action in ("HOLD", "NONE", "NEUTRAL", "NO_TRADE"):
        action = "SKIP"
    if action not in ("BUYING", "SELLING", "SKIP"):
        action = "SKIP"
    return action

_W_INDICATOR = 1.0
_W_CANDLE    = 3.5


def _blend_winrate(wr_indicator: float, wr_candle) -> float:
    if wr_candle is None or wr_candle <= 0:
        return wr_indicator
    blended = (wr_indicator * _W_INDICATOR + wr_candle * _W_CANDLE) / (_W_INDICATOR + _W_CANDLE)
    return round(blended, 4)


def analyze(pred: dict, wfv_result: dict, train_result: dict) -> dict:
    """
    Analyze signal, compute position sizing, call DeepSeek.

    wfv_result: hasil dari wfv.run_wfv() — menggantikan bt_result lama.
    """
    symbol       = pred["symbol"]
    raw_df       = train_result["raw_df"]
    wfv_after    = wfv_result["after"]
    risk_max     = RISK_PER_TRADE_PCT / 100

    # Kelly multiplier dari regime (lebih konservatif di Sideways/Volatile)
    kelly_mult = train_result.get("kelly_multiplier", 0.20)

    logger.info("[analyst] %s — regime=%s kelly_mult=%.2f",
                symbol, train_result.get("regime", "?"), kelly_mult)

    # Candle model winrate
    candle_result = train_result.get("candle_result")
    candle_bt     = candle_result.get("backtest", {}) if candle_result and candle_result.get("ok") else {}
    wr_candle_up  = candle_bt.get("winrate_up") if candle_bt else None
    wr_candle_dn  = candle_bt.get("winrate_dn") if candle_bt else None

    # Blend ML1 (WFV OOS) + ML2 (candle model)
    wr_ind_up = float(wfv_after.get("winrate_up", 0.0))
    wr_ind_dn = float(wfv_after.get("winrate_dn", 0.0))
    wr_long   = _blend_winrate(wr_ind_up, wr_candle_up)
    wr_short  = _blend_winrate(wr_ind_dn, wr_candle_dn)

    logger.info(
        "[analyst] %s WR blend — LONG: wfv=%.1f%% candle=%.1f%% blend=%.1f%% | "
        "SHORT: wfv=%.1f%% candle=%.1f%% blend=%.1f%%",
        symbol,
        wr_ind_up * 100, (wr_candle_up or 0) * 100, wr_long * 100,
        wr_ind_dn * 100, (wr_candle_dn or 0) * 100, wr_short * 100,
    )

    n_signals_long  = wfv_after.get("n_signal_up", 0) + candle_bt.get("n_signal_up", 0)
    n_signals_short = wfv_after.get("n_signal_dn", 0) + candle_bt.get("n_signal_dn", 0)

    pos_long = compute_position(
        df=raw_df,
        direction="LONG",
        winrate=wr_long,
        n_signals=n_signals_long,
        risk_per_trade=risk_max,
        max_fraction=risk_max,
        train_end=TRAIN_END_IDX,
        kelly_multiplier_override=kelly_mult,
    )
    pos_short = compute_position(
        df=raw_df,
        direction="SHORT",
        winrate=wr_short,
        n_signals=n_signals_short,
        risk_per_trade=risk_max,
        max_fraction=risk_max,
        train_end=TRAIN_END_IDX,
        kelly_multiplier_override=kelly_mult,
    )

    logger.info("[analyst] Calling DeepSeek for %s...", symbol)

    try:
        prompt = _build_prompt(pred, wfv_result, train_result, pos_long, pos_short)
        raw    = _call_deepseek(prompt)
        parsed = _parse(raw)
        action = _validate_action(parsed)
        reason = str(parsed.get("reason", ""))

        pos = pos_long if action in ("BUYING", "SKIP") else pos_short

        return {
            "ok":              True,
            "raw_response":    raw,
            "action":          action,
            "entry_price":     pos["entry_price"],
            "stop_loss":       pos["stop_loss"],
            "take_profit":     pos["take_profit"],
            "leverage":        pos["leverage"],
            "qty_fraction":    pos["qty_fraction"],
            "reason":          reason,
            "position_detail": pos,
            "regime":          train_result.get("regime", "Unknown"),
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
