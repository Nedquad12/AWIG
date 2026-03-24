"""
ai_decision.py — DeepSeek memutuskan BUY/SKIP + leverage + modal + SL/TP

Input ke AI:
  - OHLCV 20 candle terakhir
  - Semua skor indikator
  - Hasil prediksi ML (label, confidence, win_rate)
  - SL/TP suggestion dari ML (ATR-based)
  - Balance akun & posisi terbuka

Output dari AI (JSON terstruktur):
  {
    "decision":    "BUY" | "SKIP",
    "direction":   "LONG" | "SHORT",
    "leverage":    int,           (1-30)
    "capital_pct": float,         (0.05-0.25)
    "confidence":  int,           (0-100)
    "sl_pct":      float,         (% jarak SL dari entry, positif)
    "tp_pct":      float,         (% jarak TP dari entry, positif)
    "reason":      str
  }

Cara aplikasi SL/TP setelah entry:
  LONG : SL = entry * (1 - sl_pct/100),  TP = entry * (1 + tp_pct/100)
  SHORT: SL = entry * (1 + sl_pct/100),  TP = entry * (1 - tp_pct/100)
"""

import json
import logging
import re
from typing import Optional

import requests

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_URL,
    MAX_CAPITAL_PCT, MAX_LEVERAGE,
    HIGH_CONF_THRESHOLD, MID_CONF_THRESHOLD,
    SL_MIN_PCT, SL_MAX_PCT, TP_MIN_PCT, TP_MAX_PCT,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a crypto futures trading assistant. Decide whether to ENTER a trade and set Stop Loss / Take Profit.

RULES:
1. Respond ONLY in valid JSON — no markdown, no text outside the JSON object
2. Decision must be "BUY" or "SKIP"
3. If BUY: fill ALL fields — direction, leverage, capital_pct, confidence, sl_pct, tp_pct, reason
4. If SKIP: set leverage=0, capital_pct=0, confidence=0, sl_pct=0, tp_pct=0, reason (1 sentence Indonesian)

LEVERAGE & CAPITAL by confidence:
  confidence >= {high_conf} → leverage up to {max_lev}x,  capital_pct up to {max_cap}
  confidence >= {mid_conf}  → leverage up to 15x,          capital_pct up to {half_cap}
  confidence < {mid_conf}   → leverage up to 5x,           capital_pct up to {quarter_cap}

SL/TP RULES:
  - sl_pct: Stop Loss % from entry (always positive). Allowed: {sl_min}% to {sl_max}%
  - tp_pct: Take Profit % from entry (always positive). Allowed: {tp_min}% to {tp_max}%
  - Minimum Risk/Reward = 1.5  meaning tp_pct >= sl_pct * 1.5 (REQUIRED)
  - Start from the ML-suggested SL/TP baseline, you may widen but NOT tighten below {sl_min}%
  - High confidence trade: can use tighter SL closer to {sl_min}%
  - Low confidence trade: use wider SL to avoid premature stop out
  - Look at recent candle ranges to gauge volatility

OTHER RULES:
  - ML prediction NETRAL or confidence < 40: lean toward SKIP
  - Already have open position on this symbol: SKIP
  - Do not trade against clear trend (all MA aligned, RSI extreme)

Response format (STRICT JSON, nothing else):
{{
  "decision":    "BUY" or "SKIP",
  "direction":   "LONG" or "SHORT" or null,
  "leverage":    integer,
  "capital_pct": float,
  "confidence":  integer,
  "sl_pct":      float,
  "tp_pct":      float,
  "reason":      "max 2 sentences in Indonesian"
}}
""".format(
    max_lev=MAX_LEVERAGE,
    max_cap=MAX_CAPITAL_PCT,
    half_cap=round(MAX_CAPITAL_PCT * 0.5, 3),
    quarter_cap=round(MAX_CAPITAL_PCT * 0.25, 3),
    high_conf=HIGH_CONF_THRESHOLD,
    mid_conf=MID_CONF_THRESHOLD,
    sl_min=SL_MIN_PCT, sl_max=SL_MAX_PCT,
    tp_min=TP_MIN_PCT, tp_max=TP_MAX_PCT,
)


def _build_ohlcv_table(df_recent) -> str:
    lines = ["DateTime(UTC)        Open       High       Low        Close      Vol"]
    lines.append("─" * 74)
    for _, row in df_recent.iterrows():
        dt = str(row["date"])[:16]
        lines.append(
            f"{dt:<20} "
            f"{row['open']:>10.4f} {row['high']:>10.4f} "
            f"{row['low']:>10.4f} {row['close']:>10.4f} "
            f"{row['volume']:>10.0f}"
        )
    return "\n".join(lines)


def _build_prompt(
    symbol:         str,
    scores:         dict,
    ml_pred:        dict,
    ohlcv_table:    str,
    balance:        float,
    open_positions: list,
) -> str:
    has_position = any(p["symbol"] == symbol for p in open_positions)

    # ML section
    if ml_pred.get("error"):
        ml_section = f"ML Prediction: TIDAK TERSEDIA ({ml_pred['error']})"
    else:
        sl_s = ml_pred.get("sl_pct", 0.0)
        tp_s = ml_pred.get("tp_pct", 0.0)
        rr   = f"{tp_s/sl_s:.1f}x" if sl_s > 0 else "N/A"
        ml_section = (
            f"Label          : {ml_pred.get('label', '?')}  "
            f"(confidence {ml_pred.get('confidence', 0):.1f}%)\n"
            f"  Proba NAIK   : {ml_pred.get('proba_up',   0):.1f}%\n"
            f"  Proba NETRAL : {ml_pred.get('proba_flat', 0):.1f}%\n"
            f"  Proba TURUN  : {ml_pred.get('proba_down', 0):.1f}%\n"
            f"  Win Rate     : {ml_pred.get('win_rate', 0):.1f}%  "
            f"({ml_pred.get('n_train', 0)} bars)\n"
            f"  SL saran ML  : {sl_s:.2f}%  (ATR-based)\n"
            f"  TP saran ML  : {tp_s:.2f}%  (R/R {rr})"
        )

    open_pos_text = "Tidak ada" if not open_positions else ", ".join(
        f"{p['symbol']} {p['side']} qty={p['positionAmt']} entry={p['entryPrice']}"
        for p in open_positions
    )

    return (
        f"=== PAIR: {symbol} ===\n\n"
        f"ACCOUNT\n"
        f"  Balance : ${balance:.2f} USDT\n"
        f"  Posisi  : {open_pos_text}\n"
        f"  Ada {symbol}: {'YES -> pertimbangkan SKIP' if has_position else 'NO'}\n\n"
        f"INDICATOR SCORES\n"
        f"  Total  : {scores['total']:+.2f}\n"
        f"  VSA:{scores['vsa']:+d}  FSA:{scores['fsa']:+d}  VFA:{scores['vfa']:+d}  "
        f"WCC:{scores['wcc']:+d}  SRST:{scores['srst']:+d}\n"
        f"  RSI:{scores['rsi']:+d}  MACD:{scores['macd']:+d}  MA:{scores['ma']:+d}  "
        f"IP:{scores['ip_score']:+.1f}  Tight:{scores['tight']:+d}\n\n"
        f"ML PREDICTION (3 candles ahead)\n"
        f"{ml_section}\n\n"
        f"LAST 20 CANDLES (15m)\n"
        f"{ohlcv_table}\n\n"
        f"Putuskan: BUY atau SKIP? Jika BUY, tentukan leverage, capital_pct, sl_pct, tp_pct."
    )


def ask_ai(
    symbol:         str,
    scores:         dict,
    ml_pred:        dict,
    df_recent,
    balance:        float,
    open_positions: list,
) -> dict:
    """
    Kirim data ke DeepSeek, dapat keputusan trading + SL/TP.

    Returns:
        dict dengan key: decision, direction, leverage, capital_pct,
                         confidence, sl_pct, tp_pct, reason
        atau dict dengan key "error"
    """
    ohlcv_table = _build_ohlcv_table(df_recent.tail(20))
    user_prompt = _build_prompt(symbol, scores, ml_pred, ohlcv_table, balance, open_positions)

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      DEEPSEEK_MODEL,
                "max_tokens": 600,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()

        raw = (data["choices"][0]["message"].get("content") or "").strip()
        if not raw:
            raw = (data["choices"][0]["message"].get("reasoning_content") or "").strip()

        parsed = _parse_json(raw)
        if parsed is None:
            logger.error(f"[{symbol}] AI parse fail: {raw[:300]}")
            return {"error": f"Respons AI tidak valid: {raw[:200]}"}

        result = _validate(parsed)
        logger.info(
            f"[{symbol}] AI -> {result['decision']} {result.get('direction','')} | "
            f"lev={result['leverage']}x | SL={result['sl_pct']}% TP={result['tp_pct']}% | "
            f"conf={result['confidence']}%"
        )
        return result

    except requests.exceptions.Timeout:
        return {"error": "DeepSeek timeout (>90 detik)"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"DeepSeek HTTP error: {e}"}
    except Exception as e:
        logger.error(f"[{symbol}] AI error: {e}")
        return {"error": str(e)}


def _parse_json(text: str) -> Optional[dict]:
    """Coba berbagai cara parse JSON dari teks AI."""
    for attempt in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL).group(1)),
        lambda t: json.loads(re.search(r"\{.*\}", t, re.DOTALL).group(0)),
    ]:
        try:
            return attempt(text)
        except Exception:
            continue
    return None


def _validate(d: dict) -> dict:
    """Validasi dan clamp semua nilai dari respons AI."""
    decision = str(d.get("decision", "SKIP")).upper()
    if decision not in ("BUY", "SKIP"):
        decision = "SKIP"

    direction = str(d.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        direction = None

    leverage    = max(0, min(int(d.get("leverage") or 0), MAX_LEVERAGE))
    capital_pct = max(0.0, min(float(d.get("capital_pct") or 0.0), MAX_CAPITAL_PCT))
    confidence  = max(0, min(int(d.get("confidence") or 0), 100))
    sl_pct      = float(d.get("sl_pct") or 0.0)
    tp_pct      = float(d.get("tp_pct") or 0.0)

    if decision == "BUY":
        if not direction:
            decision = "SKIP"   # BUY tanpa direction -> invalid

        # Clamp SL/TP ke range config
        sl_pct = max(SL_MIN_PCT, min(sl_pct, SL_MAX_PCT))
        tp_pct = max(TP_MIN_PCT, min(tp_pct, TP_MAX_PCT))

        # Enforce minimum R/R 1.5
        if tp_pct < sl_pct * 1.5:
            tp_pct = round(sl_pct * 1.5, 2)
    else:
        sl_pct = 0.0
        tp_pct = 0.0

    return {
        "decision":    decision,
        "direction":   direction,
        "leverage":    leverage,
        "capital_pct": capital_pct,
        "confidence":  confidence,
        "sl_pct":      round(sl_pct, 2),
        "tp_pct":      round(tp_pct, 2),
        "reason":      str(d.get("reason", "")),
    }
