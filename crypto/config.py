BINANCE_API_KEY    = "v4EUTfClltUFBSOLzkL0rj5Xl9oh9OYMFBvoWvKq7rCmIrAnbpdMU895q2TbsnAc1"
BINANCE_API_SECRET = "McXPjmYSjb3yhlWfz0PH9BYrkbgLOUeSiKdibiMCaPqvM8y1ui4E20GTF0t5ywzze"
BINANCE_BASE_URL   = "https://demo-fapi.binance.com"
RECV_WINDOW        = 5000

TELEGRAM_BOT_TOKEN = "8117467835:AAFRni1PY5lmy9QC9hoh8YBjZi8RXorXCQQ"
ALLOWED_CHAT_IDS   = [6208519947, 5751902978]

DEEPSEEK_API_KEY  = "sk-85d586db34064e7b844ebf28e10794e3"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL    = "deepseek-reasoner"

INDICATOR_NAMES = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc", "funding", "lsr"]

DEFAULT_INTERVAL   = "30m"
CANDLE_LIMIT       = 1000
MIN_CANDLE_TRAIN   = 500
LOOKAHEAD          = 3
LABEL_UP_PCT       = 0.005    # +0.5% → label naik
LABEL_DOWN_PCT     = -0.005   # -0.5% → label turun
CONFIDENCE_MIN     = 0.50

DEFAULT_LEVERAGE   = 10
RISK_PER_TRADE_PCT = 1.0
WEIGHTS_DIR        = "weights"

SCHEDULE_INTERVAL_MINUTES = 30
SCAN_TOP_N         = 100
SCAN_SCORE_THRESHOLD = 1.5   # threshold realistis untuk 9 indikator

SCANNER_MAX_WORKERS  = 3     # thread paralel (rendah agar tidak kena rate limit)
SCANNER_REQUEST_DELAY = 0.5  # detik antar request dalam satu token
