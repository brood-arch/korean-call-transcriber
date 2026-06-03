"""API call logic for unified LLM extraction.

Handles HTTP requests, retry with exponential backoff,
and Langfuse observability integration.
"""

import json
import logging
import time

from src.config import get_llm_config as _resolve_llm_config
from src.pipeline.redact import redact_sensitive_text

from .prompt import get_prompt

log = logging.getLogger(__name__)

# Pipeline config
MAX_CONTENT_CHARS = 12000  # P1-4: GLM ctx 기준 여유 있음
MAX_RETRIES = 4            # increased from 3 for better resilience
RETRY_BACKOFF = [5, 15, 45, 90]  # seconds — exponential for 429


def get_llm_config(api_key: str = "") -> dict[str, str]:
    """Backward-compatible dict wrapper for shared LLM config."""
    config = _resolve_llm_config(api_key)
    return {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "model": config.model,
        "disable_thinking": config.disable_thinking,
    }


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
    resolved_base_url = (base_url or config.base_url).rstrip("/")
    resolved_model = model or config.model

    payload_obj = {
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
        # GLM coding endpoints may emit long reasoning traces by default.
        payload_obj["thinking"] = {"type": "disabled"}
    payload = json.dumps(payload_obj).encode("utf-8")

    api_url = resolved_base_url + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_key}"}

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(api_url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            text = text.strip()
            if text.startswith("```"):
                text = text.removeprefix("```json").removeprefix("```").strip()
                text = text.removesuffix("```").strip()
            return json.loads(text), result.get("usage", {})

        except json.JSONDecodeError as e:
            print(f"    Invalid JSON response: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
        except urllib.error.HTTPError as e:
            status = e.code
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception as exc:
                log.debug("Failed to read LLM HTTP error body: %s", redact_sensitive_text(repr(exc)))
            if status == 429:
                wait = 30 * (2 ** attempt)
                print(f"    429 rate limit (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            elif status >= 500:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                print(f"    Server error {status}: {redact_sensitive_text(body[:100])}, retrying in {wait}s...")
                time.sleep(wait)
            elif status == 401:
                print(f"    Auth error (key invalid?): {redact_sensitive_text(body[:100])}")
                break
            else:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                print(f"    HTTP error {status}: {redact_sensitive_text(body[:100])}, retrying in {wait}s...")
                time.sleep(wait)
        except urllib.error.URLError as e:
            wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
            print(f"    Network error: {redact_sensitive_text(repr(e))}, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    Unexpected error: {type(e).__name__}: {redact_sensitive_text(str(e))}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

    return None, {}


def call_llm_extract(api_key: str, content: str, run_id: str = "",
                     lf_available: bool = False) -> dict | None:
    """Call an OpenAI-compatible LLM with unified extraction prompt.

    Returns dict with keys: summary, todos, appointments, entities, products, money, risks, corrections
    Returns None on failure.
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

            # Langfuse: record result
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
                        }
                    )
                    _gen.end()
                except Exception as exc:
                    log.debug("Langfuse generation update failed: %s", redact_sensitive_text(repr(exc)))

            return parsed
        except Exception as e:
            print(f"    Parse error: {type(e).__name__}: {redact_sensitive_text(str(e))}")

    # Langfuse cleanup on failure
    if _gen:
        try:
            _gen.end()
        except Exception as exc:
            log.debug("Langfuse generation cleanup failed: %s", redact_sensitive_text(repr(exc)))

    return None


call_zai_extract = call_llm_extract
