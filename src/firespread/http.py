"""HTTP access with retries (urllib3 Retry) and clear errors.

Retries 429/5xx with exponential backoff; 4xx responses raise ClientError
immediately (a bad token or bad parameters won't improve by retrying).
Tests replace `get` via monkeypatch.
"""

from __future__ import annotations

import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

USER_AGENT = "firespread (UCR MS capstone; https://github.com/sharmeen-k/fire_spread_modeling)"


class ClientError(RuntimeError):
    """HTTP 4xx (other than 429)."""

    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status = status
        self.body = body


_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=2.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = USER_AGENT
        _session = s
    return _session


def get(url: str, params=None, timeout: float = 120.0) -> requests.Response:
    r = session().get(url, params=params, timeout=timeout)
    if 400 <= r.status_code < 500:
        raise ClientError(r.status_code, r.text, url)
    r.raise_for_status()
    return r


def redact(url: str, secret: str | None) -> str:
    return url.replace(secret, "***") if secret else url
