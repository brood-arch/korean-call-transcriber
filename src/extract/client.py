"""API call logic for unified LLM extraction.

Handles HTTP requests, retry with exponential backoff,
and Langfuse observability integration.
"""

from __future__ import annotations

import json
import logging
import time

from src.config import get_llm_config as _resolve_llm_config
from src.pipeline.redact import redact_sensitive_text

from .prompt import get_prompt

log = logging.getLogger(__name__)

# Pipeline config
MAX_CONTENT_CHARS = 12000  # GLM ctx 기준 여유 있음
MAX_RETRIES = 4
RETRY_BACKOFF = [5, 15, 45, 90]  # seconds — exponential for 429

# HTTP status codes that should NOT be retried
_NON_RETRYABLE = {400, 401, 403, 404}


def get_llm_config(api_key: str = "") -> dict[str, str]:
    """Backward-compatible dict wrapper for shared LLM config."""
    config = _resolve_llm_config(api_key)
    return {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "model": config.model,
        "disable_thinking": config.disable_thinking,
    }


def _extract_json_from_text(text: str) -> dict | None:
    """Extract the first JSON object from text that may contain prose or fences.

    Handles:
    - Plain JSON: ``{"key": "value"}``
    - Markdown fenced: ````json\\n{...}\\n``` ``
    - Prose prefix: ``Here is the JSON:\\n{...}``
    - Trailing text after the closing brace
    """
    text = text.strip()
    if not text:
        return None

    # Strip markdown fences
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { ... last } and try to parse
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def call_llm_json(
    prompt: str,
    *,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    max_tokens: int = 8192,
    timeout: int = 180,
    response_format: bool = True,
) -> tuple[dict | None, dict]:
    """Call an OpenAI-compatible LLM and parse JSON content from the response."""
    import urllib.error
    import urllib.request

    config = _resolve_llm_config(api_key)

    # Early validation: refuse to make HTTP requests without an API key
    if not config.api_key:
        log.error("LLM_API_KEY is not configured; skipping API call")
        return None, {"error": "missing_api_key"}

    resolved_base_url = (base_url or config.base_url).rstrip("/")
    resolved_model = model or config.model

    payload_obj: dict = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    if response_format:
        payload_obj["response_format"] = {"type": "json_object"}
    if config.disable_thinking in {"1", "true", "yes"} or (
        config.disable_thinking == "auto" and resolved_model.lower().startswith("glm")
    ):
        payload_obj["thinking"] = {"type": "disabled"}
    payload = json.dumps(payload_obj).encode("utf-8")

    api_url = resolved_base_url + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"}

    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(api_url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = _extract_json_from_text(text)
            if parsed is not None:
                return parsed, result.get("usage", {})
            log.warning("Attempt %d: could not extract JSON from LLM response", attempt + 1)
            last_error = "json_extract_failed"
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

        except json.JSONDecodeError as e:
            last_error = f"json_decode: {e}"
            log.warning("Attempt %d: JSON decode error: %s", attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

        except urllib.error.HTTPError as e:
            status = e.code
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception as exc:
                log.debug("Failed to read LLM HTTP error body: %s", redact_sensitive_text(repr(exc)))

            last_error = f"http_{status}"
            if status == 429:
                wait = 30 * (2 ** attempt)
                log.warning("429 rate limit (attempt %d/%d), waiting %ds", attempt + 1, MAX_RETRIES, wait)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
            elif status >= 500:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                log.warning("Server error %d (attempt %d/%d), retrying in %ds", status, attempt + 1, MAX_RETRIES, wait)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
            elif status in _NON_RETRYABLE:
                log.error("Non-retryable HTTP %d: %s", status, redact_sensitive_text(body[:100]))
                break
            else:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                log.warning("HTTP %d (attempt %d/%d), retrying in %ds", status, attempt + 1, MAX_RETRIES, wait)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

        except urllib.error.URLError as e:
            last_error = f"network: {e}"
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            log.warning("Network error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, redact_sensitive_text(repr(e)))
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

        except Exception as e:
            last_error = f"unexpected: {type(e).__name__}: {e}"
            log.exception("Unexpected error in LLM call (attempt %d/%d)", attempt + 1, MAX_RETRIES)
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

    log.error("All %d LLM attempts failed (last: %s)", MAX_RETRIES, last_error)
    return None, {"error": last_error, "attempts": MAX_RETRIES}


def call_llm_extract(api_key: str, content: str, run_id: str = "",
                     lf_available: bool = False) -> dict | None:
    """Call an OpenAI-compatible LLM with unified extraction prompt.

    Returns dict with keys: summary, todos, appointments, entities, products,
    money, risks, corrections.  Returns None on failure.
    """
    from .parser import parse_unified_response

    prompt_template = get_prompt()
    prompt = prompt_template.replace("{content}", content[:MAX_CONTENT_CHARS], 1)
    config = _resolve_llm_config(api_key)

    # Langfuse span
    _gen = None
    if lf_available:
        try:
            from langfuse import get_client
            _tracer = get_client()
            _gen = _tracer.start_as_current_observation(
                as_type="generation",
                name="unified-extraction-llm",
                model=config.model,
                input=prompt[:300],
                metadata={"run_id": run_id or ""},
            )
        except Exception as exc:
            log.debug("Langfuse generation setup failed: %s", redact_sensitive_text(repr(exc)))

    parsed_json, usage = call_llm_json(prompt, api_key=api_key, max_tokens=8192, timeout=180)
    if parsed_json is not None:
        try:
            text = json.dumps(parsed_json, ensure_ascii=False)
            parsed = parse_unified_response(text)
            parsed["raw_usage"] = usage

            if _gen:
                try:
                    _gen.update(
                        output={
                            "summary": bool(parsed.get("summary", {}).get("one_line")),
                            "todos": len(parsed.get("todos", [])),
                            "appointments": len(parsed.get("appointments", [])),
                            "entities": len(parsed.get("entities", [])),
                            "products": len(parsed.get("products", [])),
                            "money": len(parsed.get("money", [])),
                            "risks": len(parsed.get("risks", [])),
                            "corrections": len(parsed.get("corrections", [])),
                        },
                        metadata={
                            "prompt_tokens": parsed.get("raw_usage", {}).get("prompt_tokens", 0),
                            "completion_tokens": parsed.get("raw_usage", {}).get("completion_tokens", 0),
                        },
                    )
                    _gen.end()
                except Exception as exc:
                    log.debug("Langfuse generation update failed: %s", redact_sensitive_text(repr(exc)))

            return parsed
        except Exception as e:
            log.error("Parse error: %s: %s", type(e).__name__, redact_sensitive_text(str(e)))

    # Langfuse cleanup on failure
    if _gen:
        try:
            _gen.end()
        except Exception as exc:
            log.debug("Langfuse generation cleanup failed: %s", redact_sensitive_text(repr(exc)))

    return None


call_zai_extract = call_llm_extract
