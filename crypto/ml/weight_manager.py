"""
ml/weight_manager.py — Menyimpan dan memuat bobot indikator per ticker.

Bobot disimpan sebagai JSON di folder weights/<TICKER>.json.
Default bobot = 1.0 untuk semua indikator.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import WEIGHTS_DIR
from indicators import INDICATOR_NAMES

# Bobot default: semua 1.0
DEFAULT_WEIGHTS: dict[str, float] = {name: 1.0 for name in INDICATOR_NAMES}

# FEATURES = urutan konsisten untuk ML (sama dengan INDICATOR_NAMES)
FEATURES: list[str] = INDICATOR_NAMES


def _path(ticker: str) -> str:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, f"{ticker.upper()}.json")


def load_weights(ticker: str) -> dict[str, float]:
    """Load bobot dari file. Return DEFAULT_WEIGHTS jika belum ada."""
    p = _path(ticker)
    if not os.path.exists(p):
        return dict(DEFAULT_WEIGHTS)
    try:
        with open(p) as f:
            data = json.load(f)
        weights = dict(DEFAULT_WEIGHTS)
        weights.update({k: float(v) for k, v in data.get("weights", {}).items() if k in FEATURES})
        return weights
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def save_weights(ticker: str, weights: dict[str, float]) -> None:
    """Simpan bobot ke file dengan timestamp."""
    p = _path(ticker)
    payload = {
        "ticker":     ticker.upper(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "weights":    {k: round(float(weights.get(k, 1.0)), 6) for k in FEATURES},
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def get_weights_info(ticker: str) -> dict:
    """Return metadata bobot (updated_at, is_default)."""
    p = _path(ticker)
    if not os.path.exists(p):
        return {"updated_at": None, "is_default": True}
    try:
        with open(p) as f:
            data = json.load(f)
        return {
            "updated_at": data.get("updated_at"),
            "is_default": False,
        }
    except Exception:
        return {"updated_at": None, "is_default": True}


def apply_weights(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Hitung weighted total dari skor indikator."""
    return sum(scores.get(f, 0.0) * weights.get(f, 1.0) for f in FEATURES)
