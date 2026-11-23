"""
Centralized API Usage Optimization, Budget Control & Quota Protection Manager.
TYRION-2.0 — Final-Year Academic & Portfolio Project

Enforces strict client-side controls across the application:
1. Hard daily API call budget limits (preventing billing spikes or free quota exhaustion).
2. Per-upload maximum vision calls (preventing single documents from draining allowances).
3. Concurrency serialization (preventing concurrent API spikes triggering 429).
4. Persistent telemetry across application reboots (cached hits, errors, local vs cloud).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Optional, Tuple

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger()


class VisionQuotaManager:
    """
    Thread-safe, persistent singleton managing client-side Vision & OCR quotas.
    """

    _instance: Optional["VisionQuotaManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, state_file: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        self.state_file = state_file or (Path(getattr(settings, "VISION_CACHE_DIR", ".cache/vision")) / "usage_stats.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self._rw_lock = threading.Lock()
        self._concurrency_semaphore = threading.Semaphore(getattr(settings, "VISION_CONCURRENCY", 1))

        # Core counters
        self.daily_budget: int = getattr(settings, "VISION_DAILY_CALL_BUDGET", 50)
        self.max_calls_per_upload: int = getattr(settings, "VISION_MAX_CALLS_PER_UPLOAD", 5)

        self.last_reset_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.calls_today: int = 0
        self.current_upload_calls: int = 0
        self.images_processed: int = 0
        self.cached_hits: int = 0
        self.failed_requests: int = 0
        self.rate_limit_errors: int = 0
        self.local_ocr_calls: int = 0
        self.cloud_vision_calls: int = 0

        self._load_state()
        self._initialized = True
        logger.info(f"VisionQuotaManager ready. Daily Budget: {self.daily_budget}, Calls Today: {self.calls_today}.")

    def _check_day_rollover(self) -> None:
        """Reset daily counters if UTC calendar day has rolled over."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            logger.info(f"Vision quota daily rollover triggered ({self.last_reset_date} -> {today}). Resetting daily calls.")
            self.calls_today = 0
            self.last_reset_date = today
            self._save_state()

    def _load_state(self) -> None:
        """Load persistent telemetry from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                self.last_reset_date = data.get("last_reset_date", today)
                if self.last_reset_date == today:
                    self.calls_today = data.get("calls_today", 0)
                else:
                    self.calls_today = 0
                    self.last_reset_date = today

                self.images_processed = data.get("images_processed", 0)
                self.cached_hits = data.get("cached_hits", 0)
                self.failed_requests = data.get("failed_requests", 0)
                self.rate_limit_errors = data.get("rate_limit_errors", 0)
                self.local_ocr_calls = data.get("local_ocr_calls", 0)
                self.cloud_vision_calls = data.get("cloud_vision_calls", 0)
            except Exception as e:
                logger.warning(f"Could not load vision quota state from {self.state_file}: {e}")

    def _save_state(self) -> None:
        """Flush current telemetry to disk."""
        try:
            data = {
                "last_reset_date": self.last_reset_date,
                "calls_today": self.calls_today,
                "images_processed": self.images_processed,
                "cached_hits": self.cached_hits,
                "failed_requests": self.failed_requests,
                "rate_limit_errors": self.rate_limit_errors,
                "local_ocr_calls": self.local_ocr_calls,
                "cloud_vision_calls": self.cloud_vision_calls,
            }
            self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not save vision quota state: {e}")

    def reset_upload_session(self) -> None:
        """Reset per-upload call counter at start of a new upload batch."""
        with self._rw_lock:
            self.current_upload_calls = 0

    def can_make_cloud_call(self, for_upload: bool = True) -> Tuple[bool, str]:
        """
        Check whether client-side budget allowances permit a cloud Gemini Vision API call.
        Returns: (is_allowed: bool, reason: str)
        """
        with self._rw_lock:
            self._check_day_rollover()

            # 1. Daily configured budget ceiling
            if self.calls_today >= self.daily_budget:
                msg = f"Configured daily vision budget reached ({self.calls_today}/{self.daily_budget} calls)."
                logger.warning(msg)
                return False, msg

            # 2. Per-upload session ceiling
            if for_upload and self.current_upload_calls >= self.max_calls_per_upload:
                msg = f"Per-upload cloud vision limit reached ({self.current_upload_calls}/{self.max_calls_per_upload} calls). Remaining images processed locally."
                logger.info(msg)
                return False, msg

            return True, "OK"

    def record_call(
        self,
        provider: str,
        success: bool,
        cached: bool = False,
        rate_limited: bool = False,
    ) -> None:
        """
        Record telemetry after an image processing attempt.
        """
        with self._rw_lock:
            self._check_day_rollover()

            if cached:
                self.cached_hits += 1
                self.images_processed += 1
                self._save_state()
                return

            if rate_limited:
                self.rate_limit_errors += 1
                self.failed_requests += 1
                self._save_state()
                return

            if not success:
                self.failed_requests += 1

            if provider in ["gemini", "gemini_vision"]:
                self.cloud_vision_calls += 1
                self.calls_today += 1
                self.current_upload_calls += 1
                if success:
                    self.images_processed += 1
            elif provider in ["local_ocr", "tesseract"]:
                self.local_ocr_calls += 1
                if success:
                    self.images_processed += 1

            self._save_state()

    def get_stats(self) -> dict:
        """Retrieve aggregated API telemetry and budget diagnostics."""
        with self._rw_lock:
            self._check_day_rollover()
            remaining = max(0, self.daily_budget - self.calls_today)
            return {
                "calls_today": self.calls_today,
                "daily_budget": self.daily_budget,
                "remaining_budget": remaining,
                "images_processed": self.images_processed,
                "cached_hits": self.cached_hits,
                "failed_requests": self.failed_requests,
                "rate_limit_errors": self.rate_limit_errors,
                "local_ocr_calls": self.local_ocr_calls,
                "cloud_vision_calls": self.cloud_vision_calls,
                "current_upload_calls": self.current_upload_calls,
                "max_calls_per_upload": self.max_calls_per_upload,
                "local_first": getattr(settings, "VISION_LOCAL_FIRST", True),
                "require_user_consent": getattr(settings, "VISION_REQUIRE_USER_CONSENT", True),
                "concurrency_limit": getattr(settings, "VISION_CONCURRENCY", 1),
                "last_reset_date": self.last_reset_date,
            }

    def reset_all_counters_for_testing(self) -> None:
        """Helper to reset all counters during unit test runs."""
        with self._rw_lock:
            self.calls_today = 0
            self.current_upload_calls = 0
            self.images_processed = 0
            self.cached_hits = 0
            self.failed_requests = 0
            self.rate_limit_errors = 0
            self.local_ocr_calls = 0
            self.cloud_vision_calls = 0
            self._save_state()


def get_quota_manager() -> VisionQuotaManager:
    """Helper factory returning singleton VisionQuotaManager."""
    return VisionQuotaManager()
