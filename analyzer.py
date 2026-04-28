from pytrends.request import TrendReq
import pandas as pd
import time

from config import PYTRENDS_HL, PYTRENDS_TZ, MAX_RETRIES, BACKOFF_START, BACKOFF_RATE
from utils.logger import get_logger
from utils.cache import get_cached, set_cache

log = get_logger("analyzer")

try:
    from fake_useragent import UserAgent
    ua = UserAgent()
except ImportError:
    ua = None


class Analyzer:
    """Handles all Google Trends API calls with caching and rate-limit protection."""

    def __init__(self):
        self.pytrends = TrendReq(hl=PYTRENDS_HL, tz=PYTRENDS_TZ)
        log.info("Analyzer initialized.")

    def _refresh(self):
        """Rotate user-agent header to bypass 429 rate limits."""
        headers = {'User-Agent': ua.random} if ua else {}
        log.warning("Rotating API session (429 protection).")
        self.pytrends = TrendReq(
            hl=PYTRENDS_HL, tz=PYTRENDS_TZ,
            timeout=(10, 25),
            requests_args={'headers': headers}
        )

    def safe_call(self, func, label="call"):
        """Retry wrapper with exponential backoff."""
        for attempt in range(MAX_RETRIES):
            try:
                wait = BACKOFF_START + attempt * BACKOFF_RATE
                log.info(f"[{label}] Attempt {attempt+1}/{MAX_RETRIES} (wait {wait}s)")
                time.sleep(wait)
                result = func()
                log.info(f"[{label}] Success on attempt {attempt+1}")
                return result, None
            except Exception as e:
                if "429" in str(e):
                    log.warning(f"[{label}] Rate limited (429). Refreshing session...")
                    self._refresh()
                else:
                    log.error(f"[{label}] Failed: {e}")
                    return None, str(e)
        msg = f"[{label}] All {MAX_RETRIES} retries exhausted. Google API Rate Limit."
        log.error(msg)
        return None, msg

    def interest(self, kw: list, tf: str, geo: str):
        """Fetch interest over time. Returns cached result if available."""
        cached = get_cached("interest", str(kw), tf, geo)
        if cached is not None:
            return cached, None

        def f():
            self.pytrends.build_payload(kw, timeframe=tf, geo=geo)
            return self.pytrends.interest_over_time()

        result, err = self.safe_call(f, label="interest")
        if result is not None and not result.empty:
            set_cache("interest", result, str(kw), tf, geo)
        return result, err

    def region(self, kw: list, geo: str):
        """Fetch interest by region. Returns cached result if available."""
        cached = get_cached("region", str(kw), geo)
        if cached is not None:
            return cached, None

        def f():
            self.pytrends.build_payload(kw, geo=geo)
            return self.pytrends.interest_by_region(
                resolution='COUNTRY', inc_low_vol=False
            )

        result, err = self.safe_call(f, label="region")
        if result is not None and not result.empty:
            set_cache("region", result, str(kw), geo)
        return result, err

    def related(self, kw: list, geo: str):
        """Fetch related queries."""
        def f():
            self.pytrends.build_payload(kw, geo=geo)
            return self.pytrends.related_queries()
        return self.safe_call(f, label="related")

    def suggest(self, keyword: str):
        """Fetch Google autocomplete suggestions."""
        try:
            result = self.pytrends.suggestions(keyword)
            log.info(f"Suggestions fetched for '{keyword}'")
            return result, None
        except Exception as e:
            log.error(f"Suggestions failed for '{keyword}': {e}")
            return [], str(e)

    def realtime(self, pn: str = 'US'):
        """Fetch real-time trending searches."""
        try:
            result = self.pytrends.realtime_trending_searches(pn=pn)
            log.info(f"Realtime trends fetched for '{pn}'")
            return result, None
        except Exception as e:
            log.error(f"Realtime trends failed for '{pn}': {e}")
            return pd.DataFrame(), str(e)
