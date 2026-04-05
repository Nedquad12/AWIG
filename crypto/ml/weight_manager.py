import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_CORE: list[str] = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]
FEATURES_ALL:  list[str] = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc", "funding", "lsr"]

FEATURES: list[str] = FEATURES_CORE

try:
    from config import INDICATOR_NAMES
except ImportError:
    INDICATOR_NAMES = FEATURES_ALL

DEFAULT_WEIGHTS: dict[str, float] = {name: 1.0 for name in FEATURES_ALL}


def _path(ticker: str) -> str:
    from config import WEIGHTS_DIR
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, f"{ticker.upper()}.json")


def load_weights(ticker: str) -> dict[str, float]:
    p = _path(ticker)
    if not os.path.exists(p):
        return dict(DEFAULT_WEIGHTS)
    try:
        with open(p) as f:
            data = json.load(f)
        weights = dict(DEFAULT_WEIGHTS)
        weights.update({
            k: float(v)
            for k, v in data.get("weights", {}).items()
            if k in FEATURES_CORE
        })
        weights["funding"] = 1.0
        weights["lsr"]     = 1.0
        return weights
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def save_weights(ticker: str, weights: dict[str, float]) -> None:
    p = _path(ticker)
    payload = {
        "ticker":       ticker.upper(),
        "updated_at":   datetime.now(timezone.utc).isoformat(),
        "features":     "core_only",   # marker bahwa ini versi baru
        "weights":      {k: round(float(weights.get(k, 1.0)), 6) for k in FEATURES_CORE},
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
    """
    Hitung weighted total dari skor indikator.
    Pakai FEATURES_ALL agar funding & lsr tetap berkontribusi.
    """
    return sum(scores.get(f, 0.0) * weights.get(f, 1.0) for f in FEATURES_ALL)
