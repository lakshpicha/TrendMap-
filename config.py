# ================================================================
# config.py — Centralized configuration
# All hardcoded values live here. Edit this file to customize.
# ================================================================

import os
from dotenv import load_dotenv

load_dotenv()  # loads from .env if present

# --- Pytrends Settings ---
PYTRENDS_HL  = os.getenv("PYTRENDS_HL",  "en-US")   # Language
PYTRENDS_TZ  = int(os.getenv("PYTRENDS_TZ", 360))    # Timezone offset (India = 330, US/ET = -300)

# --- Default Search Params ---
DEFAULT_KEYWORDS = ["AI", "Cloud Computing", "Data Science"]
DEFAULT_TIMEFRAME = "today 12-m"
MAX_KEYWORDS     = 5

# --- Cache Settings ---
CACHE_DIR     = "data/cache"
CACHE_TTL_SEC = 3600  # 1 hour

# --- Logging ---
LOG_DIR   = "data/logs"
LOG_LEVEL = "INFO"

# --- Export ---
EXPORT_DIR = "data/exports"

# --- API Retry Settings ---
MAX_RETRIES   = 4
BACKOFF_START = 1   # seconds
BACKOFF_RATE  = 2   # multiplier per retry

# --- Country Map (for realtime API) ---
COUNTRY_MAP = {
    "United States":   "US",
    "India":           "IN",
    "United Kingdom":  "GB",
    "Canada":          "CA",
    "Australia":       "AU",
    "Germany":         "DE",
    "France":          "FR",
    "Brazil":          "BR",
    "Japan":           "JP",
    "Singapore":       "SG",
}
