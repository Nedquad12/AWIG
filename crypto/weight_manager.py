"""
weight_manager.py — Manajemen weight per ticker untuk scoring model

Weight disimpan per ticker di:
  /home/ec2-user/us/weights/{TICKER}.json

Format JSON:
  {
    "ticker": "AAPL",
    "updated_at": "2025-01-01T00:00:00",
    "weights": {
      "vsa": 1.0, "fsa": 1.0, "vfa": 1.0, "wcc": 1.0,
      "srst": 1.0, "rsi": 1.0, "macd": 1.0, "ma": 1.0,
      "ip_score": 1.0, "tight": 1.0
    }
  }

Jika file tidak ada → pakai DEFAULT_WEIGHTS (semua = 1.0).
Weight bersifat universal: siapapun yang update akan berlaku ke semua user.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

WEIGHTS_DIR = "/home/ec2-user/us/weights"

FEATURES = ["vsa", "fsa", "vfa", "wcc", "srst", "rsi", "macd", "ma", "ip_score", "tight"]

DEFAULT_WEIGHTS: dict[str, float] = {f: 1.0 for f in FEATURES}


def _path(ticker: str) -> str:
    return os.path.join(WEIGHTS_DIR, f"{ticker.upper()}.json")


def _ensure_dir():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def load_weights(ticker: str) -> dict[str, float]:
    """
    Baca weight untuk ticker dari JSON.
    Jika tidak ada → return DEFAULT_WEIGHTS (semua 1.0).
    """
    path = _path(ticker)
    if not os.path.exists(path):
        return dict(DEFAULT_WEIGHTS)

    try:
        with open(path, "r") as f:
            payload = json.load(f)
        w = payload.get("weights", {})
        # Pastikan semua fitur ada, fallback ke 1.0 jika ada yang kurang
        result = {}
        for feat in FEATURES:
            result[feat] = float(w.get(feat, 1.0))
        return result
    except Exception as e:
        logger.error(f"[{ticker}] Gagal baca weights: {e}, pakai default")
        return dict(DEFAULT_WEIGHTS)


def save_weights(ticker: str, weights: dict[str, float]):
    """
    Simpan weight baru ke JSON.
    """
    _ensure_dir()
    payload = {
        "ticker":     ticker.upper(),
        "updated_at": datetime.utcnow().isoformat(),
        "weights":    {f: round(float(weights.get(f, 1.0)), 6) for f in FEATURES},
    }
    path = _path(ticker)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"[{ticker}] Weights disimpan: {path}")


def apply_weights(scores: dict[str, float], weights: dict[str, float]) -> float:
    """
    Hitung total score dengan weight.

    total = sum(scores[f] * weights[f]) untuk setiap fitur

    Args:
        scores  : dict hasil indikator, key = nama fitur
        weights : dict weight per fitur

    Returns:
        float total score setelah dibobot
    """
    total = 0.0
    for feat in FEATURES:
        s = float(scores.get(feat, 0.0))
        w = float(weights.get(feat, 1.0))
        total += s * w
    return round(total, 4)


def get_weights_info(ticker: str) -> dict:
    """
    Return info weight + metadata untuk ditampilkan ke user.
    """
    path = _path(ticker)
    if not os.path.exists(path):
        return {
            "ticker":     ticker.upper(),
            "updated_at": None,
            "weights":    dict(DEFAULT_WEIGHTS),
            "is_default": True,
        }
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return {
            "ticker":     ticker.upper(),
            "updated_at": payload.get("updated_at"),
            "weights":    payload.get("weights", dict(DEFAULT_WEIGHTS)),
            "is_default": False,
        }
    except Exception:
        return {
            "ticker":     ticker.upper(),
            "updated_at": None,
            "weights":    dict(DEFAULT_WEIGHTS),
            "is_default": True,
        }


def reset_weights(ticker: str):
    """Hapus file weight → kembali ke default."""
    path = _path(ticker)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"[{ticker}] Weights direset ke default")
