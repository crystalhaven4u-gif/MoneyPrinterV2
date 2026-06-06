"""
Thin, quota-aware YouTube Data API v3 client.

This is the READ client for data farming. It uses a simple API key (from config
or the ``YOUTUBE_API_KEY`` env var) and is deliberately separate from the OAuth
upload client used for publishing.

Quota: the free tier grants 10,000 units/day. Costs per call (the ones we use):
    search.list   = 100 units
    videos.list   =   1 unit
    channels.list =   1 unit

The client tracks cumulative quota and refuses a call that would exceed a
configurable daily budget (default 9,000, leaving headroom under 10,000),
raising ``QuotaExceeded`` *before* spending the units so the farmer can stop
gracefully and persist what it already has.
"""

import time
from typing import Optional, Sequence

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"

# Quota cost in units per endpoint we call.
QUOTA_COSTS = {
    "search.list": 100,
    "videos.list": 1,
    "channels.list": 1,
}

DEFAULT_DAILY_BUDGET = 9000

# HTTP statuses worth retrying (transient).
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Indirection so tests can patch sleeping without real delays.
_sleep = time.sleep


class YouTubeAPIError(RuntimeError):
    """Raised when a request ultimately fails."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class QuotaExceeded(YouTubeAPIError):
    """Raised before a call that would exceed the configured daily budget."""


class YouTubeDataClient:
    """
    Minimal client for search.list, videos.list and channels.list with quota
    accounting and retry/backoff.
    """

    def __init__(
        self,
        api_key: str,
        daily_budget: int = DEFAULT_DAILY_BUDGET,
        session: Optional[requests.Session] = None,
        max_retries: int = 4,
        backoff_base: float = 0.5,
    ) -> None:
        if not api_key:
            raise ValueError("A YouTube Data API key is required.")
        self._api_key = api_key
        self.daily_budget = daily_budget
        self.quota_used = 0
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    @property
    def quota_remaining(self) -> int:
        return max(0, self.daily_budget - self.quota_used)

    def can_afford(self, endpoint: str) -> bool:
        """Whether another call to ``endpoint`` fits within the daily budget."""
        return self.quota_used + QUOTA_COSTS[endpoint] <= self.daily_budget

    def search_list(self, **params) -> dict:
        """Calls search.list with the given query parameters."""
        return self._request("search.list", "search", params)

    def videos_list(self, ids: Sequence[str], part: str = "contentDetails,statistics,snippet") -> dict:
        """Calls videos.list for a batch of video ids (max 50 per call)."""
        params = {"part": part, "id": ",".join(ids), "maxResults": 50}
        return self._request("videos.list", "videos", params)

    def channels_list(self, ids: Sequence[str], part: str = "statistics") -> dict:
        """Calls channels.list for a batch of channel ids (max 50 per call)."""
        params = {"part": part, "id": ",".join(ids), "maxResults": 50}
        return self._request("channels.list", "channels", params)

    def _request(self, endpoint: str, path: str, params: dict) -> dict:
        cost = QUOTA_COSTS[endpoint]

        # Stop BEFORE spending units that would breach the budget.
        if self.quota_used + cost > self.daily_budget:
            raise QuotaExceeded(
                f"Daily quota budget reached: {self.quota_used}/{self.daily_budget} "
                f"used; {endpoint} costs {cost}."
            )

        request_params = dict(params)
        request_params["key"] = self._api_key
        url = f"{API_BASE}/{path}"

        last_exception = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.request(
                    "GET", url, params=request_params, timeout=30
                )
            except requests.RequestException as exc:
                last_exception = exc
                if attempt == self._max_retries:
                    break
                _sleep(self._backoff_base * (2 ** (attempt - 1)))
                continue

            if response.status_code == 200:
                # Charge quota only on a successful call.
                self.quota_used += cost
                try:
                    return response.json()
                except ValueError as exc:
                    raise YouTubeAPIError(
                        "YouTube API returned a non-JSON response.",
                        status_code=response.status_code,
                    ) from exc

            if (
                response.status_code in RETRYABLE_STATUS_CODES
                and attempt < self._max_retries
            ):
                _sleep(self._backoff_base * (2 ** (attempt - 1)))
                continue

            raise YouTubeAPIError(
                f"YouTube API returned HTTP {response.status_code}: "
                f"{self._error_detail(response)}",
                status_code=response.status_code,
            )

        raise YouTubeAPIError(f"YouTube API request failed: {last_exception}")

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return (response.text or "").strip()[:300] or "no body"
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            return str(error.get("message", error))
        return str(payload)[:300]
