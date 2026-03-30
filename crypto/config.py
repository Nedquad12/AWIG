# =============================================================
# config.py — Konfigurasi global
# =============================================================

# --- Binance Futures DEMO (Testnet) ---
BINANCE_API_KEY    = "O5OxsMGpKbOvQHRZ8i35Ue0kwx1tZJUDYTIbymM2Djl4cIbwDgoXMNxc5VIYUkmN"
BINANCE_API_SECRET = "XQ3b1BIhMpzfC3bWWHDUTKg3AdH2YXAaToPQdEU2SUV2Sr8p7VNYWFDtMFmVfMnK"

# URL untuk ambil data market (kline, funding, lsr, ticker) — pakai produksi agar data lengkap
BINANCE_DATA_URL   = "https://fapi.binance.com"

# URL untuk eksekusi order & akun — tetap demo/testnet
BINANCE_TRADE_URL  = "https://demo-fapi.binance.com"

# Legacy alias — modul lama yang belum diupdate pakai ini (arahkan ke data)
BINANCE_BASE_URL   = BINANCE_DATA_URL

RECV_WINDOW        = 5000

# --- Telegram Bot ---
TELEGRAM_BOT_TOKEN = "8117467835:AAFRni1PY5lmy9QC9hoh8YBjZi8RXorXCQQ"
ALLOWED_CHAT_IDS   = [6208519947, 5751902978]

# --- DeepSeek API ---
DEEPSEEK_API_KEY  = "sk-85d586db34064e7b844ebf28e10794e3"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# deepseek-chat: support temperature=0, output JSON reliably tanpa "thinking" preamble
# deepseek-reasoner: sering output reasoning prose dulu baru JSON → parser sering gagal
DEEPSEEK_MODEL    = "deepseek-chat"

# --- Indikator ---
INDICATOR_NAMES = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc", "funding", "lsr"]

# --- ML & Analisis ---
DEFAULT_INTERVAL   = "30m"
CANDLE_LIMIT       = 1000
MIN_CANDLE_TRAIN   = 500
LOOKAHEAD          = 3
LABEL_UP_PCT       = 0.005
LABEL_DOWN_PCT     = -0.005
CONFIDENCE_MIN     = 0.50

# --- Order Execution ---
DEFAULT_LEVERAGE   = 10
RISK_PER_TRADE_PCT = 1.0

# Weights disimpan di path absolut agar konsisten dari manapun dijalankan
WEIGHTS_DIR        = "/home/ec2-user/crypto/weights"

# --- Scanner & Scheduler ---
SCHEDULE_INTERVAL_MINUTES = 30
SCAN_TOP_N                = 50
SCAN_SCORE_THRESHOLD      = 5

# --- Rate limiting scanner ---
SCANNER_MAX_WORKERS   = 3
SCANNER_REQUEST_DELAY = 0.5
