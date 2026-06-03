#!/usr/bin/env python3
"""
Google Calendar integration — Check today's calendar events via OAuth2.

Uses a stored OAuth2 access token (obtained separately via gcal_helper or
an OAuth2 flow) to query the Google Calendar API for today's events.

Environment variables:
    GCAL_TOKEN_PATH — Path to the OAuth2 token JSON file

The token file should contain a valid access token with calendar scope.
Token refresh/re-auth is outside the scope of this module — integrate with
your OAuth2 flow or Google's quickstart pattern as needed.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))

log = logging.getLogger(__name__)


def _get_token_path() -> Path:
    """Get the path to the Google Calendar OAuth2 token file.

    Falls back to ``state/gcal_token.json`` if GCAL_TOKEN_PATH is not set.
    """
    env_path = os.environ.get("GCAL_TOKEN_PATH", "")
    if env_path:
        return Path(env_path)
    return Path("state") / "gcal_token.json"


def _load_access_token() -> str:
    """Load the access token from the token file.

    Returns:
        Access token string.

    Raises:
        FileNotFoundError: If the token file does not exist.
        KeyError: If the token file doesn't contain an access_token field.
    """
    token_path = _get_token_path()
    if not token_path.exists():
        raise FileNotFoundError(
            f"Google Calendar token file not found: {token_path}. "
            "Set GCAL_TOKEN_PATH or run OAuth2 flow first."
        )
    data = json.loads(token_path.read_text(encoding="utf-8"))
    return data["access_token"]


def check_today() -> dict:
    """Check today's Google Calendar events.

    Returns:
        Dict with 'ok' (bool), 'count' (int), and optionally 'error' (str).
    """
    token = _load_access_token()
    today = datetime.now(KST).strftime("%Y-%m-%d")
    time_min = f"{today}T00:00:00%2B09:00"
    time_max = f"{today}T23:59:59%2B09:00"
    url = (
        f"https://www.googleapis.com/calendar/v3/calendars/primary/events"
        f"?timeMin={time_min}&timeMax={time_max}&singleEvents=true&orderBy=startTime"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        items = data.get("items", [])
        if not items:
            print("No events today")
            return {"ok": True, "count": 0}
        for item in items:
            start = item.get("start", {})
            dt = start.get("dateTime", start.get("date", "?"))
            summary = item.get("summary", "(no title)")
            print(f"  {dt} - {summary}")
        return {"ok": True, "count": len(items)}
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        log.error(f"{e.code} - {body}")
        return {"ok": False, "error": str(e)}
    except Exception as e:
        log.error(f"{e}")
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    check_today()
