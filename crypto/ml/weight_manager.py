"""
ml/weight_manager.py — Menyimpan dan memuat bobot indikator per ticker.

FIXES:
  - FEATURES_CORE: 7 indikator yang jadi fitur ML (tidak ada leakage)
  - FEATURES_ALL : 9 indikator untuk weighted scoring di predictor/scanner
  - Funding & LSR tetap berkontribusi di scoring, tapi tidak dipakai
    sebagai fitur ML karena tidak ada historical per-candle data untuk keduanya.

Bobot disimpan sebagai JSON di folder weights/<TICKER>.json.
Default bobot = 1.0 untuk semua indikator.
"""

import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Hardcoded — tidak bergantung pada config agar tidak ada circular import
# atau KeyError jika config lama belum di-update
FEATURES_CORE: list[str] = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc"]
FEATURES_ALL:  list[str] = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc", "funding", "lsr"]

# Alias backward-compat — kode lama yang import FEATURES masih jalan
FEATURES: list[str] = FEATURES_CORE

try:
    from config import INDICATOR_NAMES
except ImportError:
    INDICATOR_NAMES = FEATURES_ALL

# Default bobot: semua 1.0
DEFAULT_WEIGHTS: dict[str, float] = {name: 1.0 for name in FEATURES_ALL}


def _path(ticker: str) -> str:
    from config import WEIGHTS_DIR
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, f"{ticker.upper()}.json")


def load_weights(ticker: str) -> dict[str, float]:
    """
    Load bobot dari file.
    Return DEFAULT_WEIGHTS jika belum ada.
    Funding & LSR selalu punya bobot default 1.0 karena tidak di-train.
    """
    p = _path(ticker)
    if not os.path.exists(p):
        return dict(DEFAULT_WEIGHTS)
    try:
        with open(p) as f:
            data = json.load(f)
        weights = dict(DEFAULT_WEIGHTS)
        # Update hanya FEATURES_CORE dari file (yang di-train)
        weights.update({
            k: float(v)
            for k, v in data.get("weights", {}).items()
            if k in FEATURES_CORE
        })
        # funding & lsr selalu 1.0 — tidak di-train, tapi tetap masuk scoring
        weights["funding"] = 1.0
        weights["lsr"]     = 1.0
        return weights
    except Exception:
        return dict(DEFAULT_WEIGHTS)


def save_weights(ticker: str, weights: dict[str, float]) -> None:
    """
    Simpan bobot ke file dengan timestamp.
    Hanya simpan FEATURES_CORE — funding & lsr tidak di-train.
    """
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
