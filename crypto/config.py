BINANCE_API_KEY    = "qzzmHPXPtrL"
BINANCE_API_SECRET = "WES4z7IvpwQ010hTfU30"
BINANCE_DATA_URL   = "https://fapi.binance.com"
BINANCE_TRADE_URL  = "https://fapi.binance.com"

BINANCE_BASE_URL   = BINANCE_DATA_URL

RECV_WINDOW        = 5000

TELEGRAM_BOT_TOKEN = "8117467835:AAXorXCQQ"
ALLOWED_CHAT_IDS   = [6208519947, 5751902978]

DEEPSEEK_API_KEY  = "skc"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEEPSEEK_MODEL    = "deepseek-chat"

INDICATOR_NAMES = ["vsa", "fsa", "vfa", "rsi", "macd", "ma", "wcc", "funding", "lsr"]

DEFAULT_INTERVAL   = "4h"
CANDLE_LIMIT       = 405
MIN_CANDLE_TRAIN   = 300
LOOKAHEAD          = 3
LABEL_UP_PCT       = 0.008
LABEL_DOWN_PCT     = -0.008
CONFIDENCE_MIN     = 0.50

DEFAULT_LEVERAGE   = 20
RISK_PER_TRADE_PCT = 3.0

WEIGHTS_DIR        = "/home/ec2-user/crypto/weights"

SCHEDULE_INTERVAL_MINUTES = 240
SCAN_TOP_N                = 50
SCAN_SCORE_THRESHOLD      = 4.5

SCANNER_MAX_WORKERS   = 3
SCANNER_REQUEST_DELAY = 0.5

# ------------------------------------------------------------------
# Paper Trading
# ------------------------------------------------------------------
PAPER_TRADING_MODE  = True       # ← ganti False kalau mau real trading
PAPER_BALANCE_USDT  = 5000.0     # virtual balance untuk hitung sizing

SCANNER_BLACKLIST = {
    "BTCUSDT",
    "BUSDUSDT",
    "USDCUSDT",
    "TUSDUSDT",
    "FDUSDUSDT",
}
