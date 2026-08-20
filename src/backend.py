"""Async OpenRouter chat backend with an on-disk response cache.

Every call is keyed by (model, messages, sampling params). Results are appended
to a JSONL cache so that interrupted runs resume for free and re-analysis never
re-spends tokens.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import httpx

from config import CACHE_PATH

API_URL = "https://openrouter.ai/api/v1/chat/completions"

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _key_from_dotenv() -> str | None:
    """Read OPENROUTER_API_KEY from a .env at the project root, if present."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPENROUTER_API_KEY"):
            _, _, val = line.partition("=")
            return val.strip().strip("'\"") or None
    return None


def _key(model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    blob = json.dumps(
        {"m": model, "msgs": messages, "mt": max_tokens, "t": temperature},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class Backend:
    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        concurrency: int = 16,
        max_retries: int = 10,
        timeout: float = 120.0,
        throttle_budget: float = 7200.0,
    ):
        self.api_key = os.environ.get("OPENROUTER_API_KEY") or _key_from_dotenv()
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it or put it in a .env file."
            )
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        self._load_cache()
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.timeout = timeout
        # Wall-clock seconds a single call may spend waiting out HTTP 429s before
        # they are treated as a real failure.
        self.throttle_budget = throttle_budget
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None
        self.n_api_calls = 0
        self.n_cache_hits = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # -- cache -----------------------------------------------------------------
    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self.cache[rec["k"]] = rec["v"]
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a truncated final line

    async def _cache_put(self, k: str, v: str) -> None:
        async with self._lock:
            self.cache[k] = v
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")

    # -- lifecycle -------------------------------------------------------------
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    # -- main entry point ------------------------------------------------------
    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        k = _key(model, messages, max_tokens, temperature)
        if k in self.cache:
            self.n_cache_hits += 1
            return self.cache[k]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with self.sem:
            text = await self._post_with_retries(payload)
        await self._cache_put(k, text)
        return text

    async def _post_with_retries(self, payload: dict) -> str:
        assert self._client is not None
        last_err = None
        attempt = 0
        deadline = time.monotonic() + self.throttle_budget
        while attempt < self.max_retries:
            try:
                r = await self._client.post(API_URL, json=payload)
                if r.status_code in _RETRY_STATUS:
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    # A 429 is upstream throttling, not a failed request: on free-tier
                    # models it can persist for many minutes and applies to every call
                    # for that model, so spending the retry budget on it would abort a
                    # multi-hour run over a transient condition. Keep waiting against a
                    # wall-clock deadline instead, without consuming attempts.
                    if r.status_code == 429 and time.monotonic() < deadline:
                        await self._sleep_backoff(self.max_retries, r)
                        continue
                    attempt += 1
                    await self._sleep_backoff(attempt - 1, r)
                    continue
                r.raise_for_status()
                data = r.json()

                if "error" in data and not data.get("choices"):
                    last_err = f"API error: {str(data['error'])[:200]}"
                    attempt += 1
                    await self._sleep_backoff(attempt - 1)
                    continue

                usage = data.get("usage") or {}
                self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += usage.get("completion_tokens", 0) or 0
                self.n_api_calls += 1

                msg = data["choices"][0].get("message") or {}
                return (msg.get("content") or "").strip()
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
                last_err = f"{type(e).__name__}: {e}"
                attempt += 1
                await self._sleep_backoff(attempt - 1)
        raise RuntimeError(
            f"OpenRouter call failed after {self.max_retries} attempts "
            f"(model={payload['model']}): {last_err}"
        )

    async def _sleep_backoff(self, attempt: int, resp: httpx.Response | None = None) -> None:
        delay = None
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    delay = float(ra)
                except ValueError:
                    delay = None
        if delay is None:
            # Upstream free-tier rate limits can persist for minutes, so cap the
            # backoff high enough that the retry budget spans one rather than 60s.
            delay = min(2.0 ** attempt, 60.0) * (0.5 + random.random())
        await asyncio.sleep(delay)

    def stats(self) -> dict:
        return {
            "api_calls": self.n_api_calls,
            "cache_hits": self.n_cache_hits,
            **self.usage,
        }
