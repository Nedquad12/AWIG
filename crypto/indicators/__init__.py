# =============================================================
# indicators/__init__.py
# INDICATOR_NAMES diimport dari config agar selalu sinkron
# =============================================================

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import INDICATOR_NAMES

from .vsa  import score_vsa,  analyze as analyze_vsa
from .rsi  import score_rsi,  analyze as analyze_rsi
from .macd import score_macd, analyze as analyze_macd
from .ma   import score_ma,   analyze as analyze_ma
from .fsa  import score_fsa,  analyze as analyze_fsa
from .vfa  import score_vfa,  get_vfa_detail, analyze as analyze_vfa
from .wcc  import score_wcc,  get_wcc_detail, analyze as analyze_wcc

__all__ = [
    "INDICATOR_NAMES",
    "score_vsa",  "analyze_vsa",
    "score_rsi",  "analyze_rsi",
    "score_macd", "analyze_macd",
    "score_ma",   "analyze_ma",
    "score_fsa",  "analyze_fsa",
    "score_vfa",  "get_vfa_detail", "analyze_vfa",
    "score_wcc",  "get_wcc_detail", "analyze_wcc",
]
