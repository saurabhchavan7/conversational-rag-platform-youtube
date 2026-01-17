import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Test different log levels
logger.debug("This is a DEBUG message (only shows if LOG_LEVEL=DEBUG)")
logger.info("✅ This is an INFO message")
logger.warning("⚠️ This is a WARNING message")
logger.error("❌ This is an ERROR message")
logger.critical("🔥 This is a CRITICAL message")

print("\n" + "="*60)
print("Check logs/app.log file to see file logging!")
print("="*60)