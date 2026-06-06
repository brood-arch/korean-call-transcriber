"""API call logic for unified LLM extraction.

Handles HTTP requests, retry with exponential backoff,
and Langfuse observability integration.
"""

from __future__ import annotations

import json
import logging
import time
import warnings as _w
from typing import Any

from kct.config import get_llm_config as _resolve_llm_config
from kct.pipeline.redact import redact_sensitive_text

from .prompt import get_prompt

log = logging.getLogger(__name__)

# Pipeline config
MAX_CONTENT_CHARS = 12000  # GLM ctx 기준 여유 있음
MAX_RETRIES = 4
RETRY_BACKOFF = [5, 15, 45, 90]  # seconds — exponential for 429

# HTTP status codes that should NOT be retried
_NON_RETRYABLE = {400, 401, 403, 404}


def get_llm_config(api_key: str = "") -> dict[str, str]:
    """공유 LLM 설정을 딕셔너리로 반환하는 하위호환 래퍼."""
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
        pass  # Expected: text may contain prose before/after JSON

    # Find first { ... last } and try to parse
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            log.debug("Failed to parse extracted JSON object: %s", exc)

    return None


def _build_llm_request(
    prompt: str, config: Any, base_url: str, model: str,
    max_tokens: int, response_format: bool, timeout: int,
) -> Any:
    """Build the HTTP request for an OpenAI-compatible LLM call."""
    import urllib.request

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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    return urllib.request.Request(api_url, data=payload, headers=headers)


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
    """OpenAI 호환 LLM을 호출하고 응답에서 JSON을 파싱한다."""
    config = _resolve_llm_config(api_key)

    # Early validation: refuse to make HTTP requests without an API key
    if not config.api_key:
        log.error("LLM_API_KEY is not configured; skipping API call")
        return None, {"error": "missing_api_key"}

    req = _build_llm_request(
        prompt, config, base_url, model, max_tokens, response_format, timeout,
    )

    last_error = ""
    for attempt in range(MAX_RETRIES):
        parsed, usage, err = _attempt_llm_call(req, timeout, attempt)
        if err is None:
            return parsed, usage
        if err == "_break":
            break
        last_error = err

    log.error("All %d LLM attempts failed (last: %s)", MAX_RETRIES, last_error)
    return None, {"error": last_error, "attempts": MAX_RETRIES}


def _attempt_llm_call(req: Any, timeout: int, attempt: int) -> tuple[dict | None, dict | None, str | None]:
    """Execute one attempt of the LLM call.

    Returns (parsed, usage, error_str).
    error_str is None on success, or "_break" to stop retrying.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return _handle_http_error_attempt(e, attempt)
    except urllib.error.URLError as e:
        return _handle_url_error(e, attempt)
    except json.JSONDecodeError as e:
        return _simple_retry(f"json_decode: {e}", 5, attempt)
    except (ConnectionError, TimeoutError, RuntimeError) as e:  # noqa: BLE001
        log.exception("Unexpected error in LLM call (attempt %d/%d)", attempt + 1, MAX_RETRIES)
        return _simple_retry(f"unexpected: {type(e).__name__}: {e}", 5, attempt)

    text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _extract_json_from_text(text)
    if parsed is not None:
        return parsed, result.get("usage", {}), None
    log.warning("Attempt %d: could not extract JSON from LLM response", attempt + 1)
    if attempt < MAX_RETRIES - 1:
        time.sleep(5)
    return None, None, "json_extract_failed"


def _handle_http_error_attempt(e: Any, attempt: int) -> tuple[None, None, str | None]:
    """Handle HTTPError for a single attempt. Returns (None, None, error_or_break)."""
    status = e.code
    body = ""
    try:
        body = e.read().decode("utf-8", errors="replace")[:200]
    except (OSError, UnicodeDecodeError) as exc:  # noqa: BLE001
        log.debug("Failed to read LLM HTTP error body: %s", redact_sensitive_text(repr(exc)))

    if status == 429:
        wait = 30 * (2 ** attempt)
        log.warning("429 rate limit (attempt %d/%d), waiting %ds", attempt + 1, MAX_RETRIES, wait)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait)
        return None, None, f"http_{status}"

    if status >= 500:
        wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
        log.warning("Server error %d (attempt %d/%d), retrying in %ds", status, attempt + 1, MAX_RETRIES, wait)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait)
        return None, None, f"http_{status}"

    if status in _NON_RETRYABLE:
        log.error("Non-retryable HTTP %d: %s", status, redact_sensitive_text(body[:100]))
        return None, None, "_break"

    wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
    log.warning("HTTP %d (attempt %d/%d), retrying in %ds", status, attempt + 1, MAX_RETRIES, wait)
    if attempt < MAX_RETRIES - 1:
        time.sleep(wait)
    return None, None, f"http_{status}"


def _handle_url_error(e: Any, attempt: int) -> tuple[None, None, str]:
    """Handle URLError for a single attempt. Returns (None, None, error_str)."""
    wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
    log.warning(
        "Network error (attempt %d/%d): %s",
        attempt + 1, MAX_RETRIES, redact_sensitive_text(repr(e)),
    )
    if attempt < MAX_RETRIES - 1:
        time.sleep(wait)
    return None, None, f"network: {e}"


def _simple_retry(error_str: str, backoff: int, attempt: int) -> tuple[None, None, str]:
    """Sleep and return a retryable error. Returns (None, None, error_str)."""
    if attempt < MAX_RETRIES - 1:
        time.sleep(backoff)
    return None, None, error_str


def call_llm_extract(api_key: str, content: str, run_id: str = "",
                     lf_available: bool = False) -> dict | None:
    """통합 추출 프롬프트로 LLM을 호출해 결과 딕셔너리를 반환한다."""
    # Returns dict with keys: summary, todos, appointments, entities, products,
    # money, risks, corrections.  Returns None on failure.
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
        except (ValueError, KeyError, TypeError) as e:
            log.error("Parse error: %s: %s", type(e).__name__, redact_sensitive_text(str(e)))

    # Langfuse cleanup on failure
    if _gen:
        try:
            _gen.end()
        except Exception as exc:
            log.debug("Langfuse generation cleanup failed: %s", redact_sensitive_text(repr(exc)))

    return None


def call_zai_extract(*args, **kwargs) -> dict | None:
    """call_llm_extract의 deprecated 별칭."""
    _w.warn("call_zai_extract is deprecated; use call_llm_extract", DeprecationWarning, stacklevel=2)
    return call_llm_extract(*args, **kwargs)
