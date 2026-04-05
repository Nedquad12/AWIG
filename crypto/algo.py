import logging
from telegram_bot import build_application

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    logger.info("🚀 Starting Binance Futures Demo Bot...")
    app = build_application()
    logger.info("✅ Bot aktif. Tekan Ctrl+C untuk berhenti.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
