"""Google Gemini API Provider for Harvey.

Allows Harvey to run autonomous sales cycles using Google Gemini
(e.g., gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash) via a standard
GEMINI_API_KEY without requiring external subscription CLI tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("harvey.gemini")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiError(Exception):
    """Base error for Gemini API calls."""
    def __init__(self, message: str, retryable: bool = False, status_code: int = 0):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class GeminiClient:
    """Async client for Google Gemini API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model = model or os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL

    def is_configured(self) -> bool:
        """Check if GEMINI_API_KEY is available."""
        return bool(self.api_key)

    def _sync_request(self, payload: dict, timeout: float) -> dict:
        """Execute synchronous HTTP request to Gemini API."""
        if not self.api_key:
            raise GeminiError(
                "GEMINI_API_KEY is not set. Add GEMINI_API_KEY=your_key to your .env file or environment.",
                retryable=False,
            )

        endpoint = f"{GEMINI_API_BASE}/{self.model}:generateContent?key={self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            status = e.code
            retryable = status in (429, 500, 503, 504)
            msg = f"Gemini API HTTP {status}: {err_body[:300]}"
            if status == 400:
                logger.error(f"Gemini bad request: {err_body}")
            elif status == 403 or status == 401:
                logger.error("Gemini authentication failed. Verify GEMINI_API_KEY in .env.")
            elif status == 429:
                logger.warning("Gemini rate limit exceeded. Backing off...")
            raise GeminiError(msg, retryable=retryable, status_code=status) from e
        except urllib.error.URLError as e:
            raise GeminiError(f"Network error reaching Gemini: {e.reason}", retryable=True) from e
        except Exception as e:
            raise GeminiError(f"Unexpected error calling Gemini: {e}", retryable=False) from e

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        expect_json: bool = False,
        timeout: float = 60.0,
    ) -> str:
        """Generate text from Gemini asynchronously."""
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2 if expect_json else 0.4,
                "maxOutputTokens": 4096,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if expect_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        # Offload network I/O to thread pool
        resp_data = await asyncio.to_thread(self._sync_request, payload, timeout)

        try:
            candidates = resp_data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini returned empty candidates list.")
                return ""

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return ""

            text_parts = [p.get("text", "") for p in parts if "text" in p]
            return "".join(text_parts).strip()
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return ""


class GeminiQuotaClient:
    """Mock/Passthrough quota manager for Gemini API (free tier & pay-as-you-go)."""

    def __init__(self, client: GeminiClient):
        self.client = client

    async def check_usage(self) -> float:
        """Gemini pay-per-token usage is budget-friendly. Return current usage ratio."""
        return 0.05
