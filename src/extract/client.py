"""API call logic for unified LLM extraction.

Handles HTTP requests, retry with exponential backoff,
and Langfuse observability integration.
"""

import json
import os
import time

from .prompt import get_prompt

# Pipeline config
MAX_CONTENT_CHARS = 12000  # P1-4: GLM ctx 기준 여유 있음
MAX_RETRIES = 4            # increased from 3 for better resilience
RETRY_BACKOFF = [5, 15, 45, 90]  # seconds — exponential for 429


def get_llm_config(api_key: str = "") -> dict[str, str]:
    """Resolve OpenAI-compatible LLM settings.

    LLM_* variables are preferred for public use. ZAI_* variables are accepted
    for backward compatibility with the original deployment.
    """
    return {
        "api_key": api_key or os.environ.get("LLM_API_KEY") or os.environ.get("ZAI_API_KEY", ""),
        "base_url": (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("ZAI_BASE_URL")
            or "https://api.z.ai/api/coding/paas/v4"
        ).rstrip("/"),
        "model": os.environ.get("LLM_MODEL", "glm-5.1"),
    }


def call_llm_extract(api_key: str, content: str, run_id: str = "",
                     lf_available: bool = False) -> dict | None:
    """Call an OpenAI-compatible LLM with unified extraction prompt.

    Returns dict with keys: summary, todos, appointments, entities, products, money, risks, corrections
    Returns None on failure.
    """
    import urllib.error
    import urllib.request

    from .parser import parse_unified_response

    prompt_template = get_prompt()
    prompt = prompt_template.replace("{content}", content[:MAX_CONTENT_CHARS], 1)
    config = get_llm_config(api_key)

    payload_obj = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    disable_thinking = os.environ.get("LLM_DISABLE_THINKING", "auto").lower()
    if disable_thinking in {"1", "true", "yes"} or (
        disable_thinking == "auto" and config["model"].lower().startswith("glm")
    ):
        # GLM coding endpoints may emit long reasoning traces by default.
        payload_obj["thinking"] = {"type": "disabled"}
    payload = json.dumps(payload_obj).encode("utf-8")

    api_url = config["base_url"] + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config['api_key']}"}

    # Langfuse span
    _gen = None
    if lf_available:
        try:
            from langfuse import get_client
            _tracer = get_client()
            _gen = _tracer.start_as_current_observation(
                as_type="generation",
                name="unified-extraction-llm",
                model=config["model"],
                input=prompt[:300],
                metadata={"run_id": run_id or ""},
            )
        except Exception:
            pass

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(api_url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_unified_response(text)
            parsed["raw_usage"] = result.get("usage", {})

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
                except Exception:
                    pass

            return parsed

        except urllib.error.HTTPError as e:
            status = e.code
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            if status == 429:
                # Exponential backoff: 30s, 60s, 120s
                wait = 30 * (2 ** attempt)
                print(f"    429 rate limit (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            elif status >= 500:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                print(f"    Server error {status}: {body[:100]}, retrying in {wait}s...")
                time.sleep(wait)
            elif status == 401:
                print(f"    Auth error (key invalid?): {body[:100]}")
                break  # No point retrying with bad key
            else:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                print(f"    HTTP error {status}: {body[:100]}, retrying in {wait}s...")
                time.sleep(wait)
        except urllib.error.URLError as e:
            wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
            print(f"    Network error: {e}, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    Unexpected error: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            # Don't break — try remaining attempts

    # Langfuse cleanup on failure
    if _gen:
        try:
            _gen.end()
        except Exception:
            pass

    return None


call_zai_extract = call_llm_extract
