"""
config.py — Konfigurasi utama crypto trading bot
"""

# ── Binance Futures Testnet ───────────────────────────────────────────────────
BINANCE_API_KEY    = "dokF6vEYW3FWV4flz8fZXxPIZVZkWD0tOo5FsMQ1azCQRZZmOoQPUlisgDEQJ52w"
BINANCE_API_SECRET = "edospwrqrJe4r1WpM5Pcn60eAIE1h5FmpguS73ROaQmO6dGnVH6AbqDLLTZ5XFNg"
BINANCE_BASE_URL   = "https://testnet.binancefuture.com"   # Futures Testnet

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8117467835:AAFoQROc4UrvUPhldXgZlDZ_kJduSLB2GGI"
TELEGRAM_CHAT_ID   = -1001855219301        # chat_id / group_id untuk notifikasi
TELEGRAM_TOPIC_ID  = None       # None jika bukan supergroup dengan topic

# ── DeepSeek ──────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY   = "sk-a13eee8d48d7475b97c2bf14db79423c"
DEEPSEEK_MODEL     = "deepseek-reasoner"
DEEPSEEK_URL       = "https://api.deepseek.com/v1/chat/completions"

# ── Pair yang di-trade ────────────────────────────────────────────────────────
# Tambah/hapus pair di sini. Bot akan cek semua pair setiap 15 menit.
TRADE_PAIRS = [
    "BNBUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

# ── Timeframe ─────────────────────────────────────────────────────────────────
TIMEFRAME          = "15m"
CANDLE_LIMIT       = 500        # jumlah candle yang di-fetch (max Binance = 1500)
SCORE_WARMUP       = 200        # bar warmup untuk MA200

# ── Risk Management ───────────────────────────────────────────────────────────
MAX_CAPITAL_PCT    = 0.25       # max 25% modal per trade
MAX_LEVERAGE       = 30         # max leverage
MIN_LEVERAGE       = 1

# Confidence threshold untuk leverage scaling
# confidence dari DeepSeek (0-100):
#   >= HIGH_CONF  → pakai modal & leverage penuh
#   >= MID_CONF   → pakai 50%
#   < MID_CONF    → pakai 25%
HIGH_CONF_THRESHOLD = 70
MID_CONF_THRESHOLD  = 50

# ── Stop Loss / Take Profit ───────────────────────────────────────────────────
# Semua nilai dalam % dari entry price (positif)
SL_MIN_PCT   = 0.5     # SL minimum 0.5% dari entry
SL_MAX_PCT   = 5.0     # SL maksimum 5% dari entry
TP_MIN_PCT   = 0.75    # TP minimum 0.75% dari entry
TP_MAX_PCT   = 15.0    # TP maksimum 15% dari entry

# SL/TP suggestion dari ML (ATR-based)
SL_ATR_MULT  = 1.5     # SL = ATR14 * 1.5  (dalam %)
TP_RR_RATIO  = 2.0     # TP = SL * 2.0     (Risk/Reward default)

# ── Indikator S&R ─────────────────────────────────────────────────────────────
SR_METHOD          = "Donchian"
SR_SENSITIVITY     = 10
SR_ATR_PERIOD      = 200
SR_ATR_MULT        = 0.5
SR_MAX_LEVELS      = 5

# ── Tight scan ────────────────────────────────────────────────────────────────
TIGHT_MA_PERIODS   = [3, 5, 10, 20]
VT_THRESHOLD       = 5.0
T_MIN              = 5.0
T_MAX              = 15.0

# ── Paths ─────────────────────────────────────────────────────────────────────
import os
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR          = os.path.join(BASE_DIR, "cache")
WEIGHTS_DIR        = os.path.join(BASE_DIR, "weights")
HISTORY_DIR        = os.path.join(BASE_DIR, "history")
LOG_DIR            = os.path.join(BASE_DIR, "logs")

# ── Trade state ───────────────────────────────────────────────────────────────
# File JSON untuk simpan posisi yang sedang buka
POSITIONS_FILE     = os.path.join(BASE_DIR, "positions.json")
