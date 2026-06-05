"""JSON response parsing and field validation for unified extraction.

Handles markdown code block stripping, JSON extraction from mixed text,
and per-category field validation with type coercion.
"""

import json
import logging

log = logging.getLogger(__name__)


def parse_unified_response(text: str) -> dict:
    """Parse unified JSON response, handling markdown code blocks."""
    cleaned = text.strip()

    # Strip markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
        return _build_result(data)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
                return _build_result(data)
            except json.JSONDecodeError as exc:
                log.debug("Failed to parse extracted JSON object: %s", exc)
        return {
            "summary": {}, "todos": [], "appointments": [], "entities": [],
            "products": [], "money": [], "risks": [], "corrections": [],
            "parse_error": True, "raw": cleaned[:500]
        }


def _build_result(data: dict) -> dict:
    """Build validated result dict from parsed JSON data."""
    return {
        "summary": _validate_summary(data.get("summary", {})),
        "todos": _validate_todos(data.get("todos", [])),
        "appointments": _validate_appointments(data.get("appointments", [])),
        "entities": _validate_entities(data.get("entities", [])),
        "products": _validate_products(data.get("products", [])),
        "money": _validate_money(data.get("money", [])),
        "risks": _validate_risks(data.get("risks", [])),
        "corrections": _validate_corrections(data.get("corrections", [])),
        "parse_error": False,
    }


def _validate_summary(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    return {
        "one_line": str(item.get("one_line", ""))[:300],
        "details": [str(d)[:200] for d in item.get("details", []) if d][:5],
        "call_type": str(item.get("call_type", "unknown")),
        "overall_confidence": float(item.get("overall_confidence", 0.0)),
    }


def _validate_todos(items: list) -> list:
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("title"):
            result.append({
                "id_hint": str(item.get("id_hint", ""))[:50],
                "title": str(item["title"]).strip(),
                "owner": (
                    item.get("owner", "unknown")
                    if item.get("owner") in ("me", "partner", "unknown")
                    else "unknown"
                ),
                "priority": (
                    item.get("priority", "medium")
                    if item.get("priority") in ("high", "medium", "low")
                    else "medium"
                ),
                "status": (
                    item.get("status", "new")
                    if item.get("status") in ("new", "in_progress", "waiting", "done", "cancelled")
                    else "new"
                ),
                "due_date": str(item.get("due_date")) if item.get("due_date") else None,
                "due_time": str(item.get("due_time")) if item.get("due_time") else None,
                "context": str(item.get("context", ""))[:200],
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_appointments(items: list) -> list:
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("title"):
            result.append({
                "title": str(item["title"]).strip(),
                "date": str(item.get("date")) if item.get("date") else None,
                "time": str(item.get("time")) if item.get("time") else None,
                "timezone": str(item.get("timezone", "Asia/Seoul")),
                "location": str(item.get("location")) if item.get("location") else None,
                "participants": [str(p) for p in item.get("participants", []) if p][:10],
                "description": str(item.get("description", ""))[:200],
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_entities(items: list) -> list:
    result = []
    valid_types = {
        "Person", "Organization", "Location", "PhoneNumber",
        "Product", "Project", "Event", "Contract", "Other"
    }
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            etype = item.get("type", "")
            if etype not in valid_types:
                etype = "Other"
            result.append({
                "name": str(item["name"]).strip(),
                "type": etype,
                "canonical_name": str(item.get("canonical_name")) if item.get("canonical_name") else None,
                "role": (
                    item.get("role", "unknown")
                    if item.get("role") in ("customer", "supplier", "employee", "carrier", "unknown")
                    else "unknown"
                ),
                "attributes": (
                    item.get("attributes", {})
                    if isinstance(item.get("attributes"), dict)
                    else {}
                ),
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_products(items: list) -> list:
    result = []
    valid_cats = {"환풍기", "송풍기", "케이스", "날개", "채반", "부품", "기타", "unknown"}
    valid_actions = {"quote", "order", "deliver", "repair", "check_stock", "manufacture", "unknown"}
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            qty = item.get("quantity", {})
            if not isinstance(qty, dict):
                qty = {}
            result.append({
                "name": str(item["name"]).strip(),
                "canonical_name": str(item.get("canonical_name")) if item.get("canonical_name") else None,
                "category": item.get("category", "unknown") if item.get("category") in valid_cats else "unknown",
                "spec": str(item.get("spec")) if item.get("spec") else None,
                "quantity": {
                    "value": int(qty.get("value", 0)),
                    "unit": str(qty.get("unit", "unknown")),
                },
                "action": item.get("action", "unknown") if item.get("action") in valid_actions else "unknown",
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_money(items: list) -> list:
    result = []
    valid_kinds = {"price", "deposit", "balance", "shipping", "discount", "tax", "unknown"}
    valid_statuses = {"paid", "unpaid", "partial", "unknown"}
    for item in items:
        if isinstance(item, dict):
            try:
                amount = int(item.get("amount", 0))
            except (ValueError, TypeError):
                amount = 0
            result.append({
                "amount": amount,
                "currency": str(item.get("currency", "KRW")),
                "kind": (
                    item.get("kind", "unknown")
                    if item.get("kind") in valid_kinds
                    else "unknown"
                ),
                "related_to": (
                    str(item.get("related_to"))
                    if item.get("related_to") else None
                ),
                "payment_status": (
                    item.get("payment_status", "unknown")
                    if item.get("payment_status") in valid_statuses
                    else "unknown"
                ),
                "due_date": str(item.get("due_date")) if item.get("due_date") else None,
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_risks(items: list) -> list:
    result = []
    valid_severities = {"high", "medium", "low"}
    valid_types = {
        "missed_deadline", "payment_delay", "customer_complaint",
        "stock_shortage", "quality_issue", "privacy",
        "ambiguous_request", "other"
    }
    for item in items:
        if isinstance(item, dict) and item.get("description"):
            result.append({
                "severity": item.get("severity", "medium") if item.get("severity") in valid_severities else "medium",
                "type": item.get("type", "other") if item.get("type") in valid_types else "other",
                "description": str(item["description"])[:300],
                "recommended_action": str(item.get("recommended_action")) if item.get("recommended_action") else None,
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_corrections(items: list) -> list:
    result = []
    valid_reasons = {"exact_rule", "alias", "contextual", "spacing", "number_normalization", "other"}
    for item in items:
        if isinstance(item, dict) and item.get("original") and item.get("corrected"):
            result.append({
                "original": str(item["original"])[:200],
                "corrected": str(item["corrected"])[:200],
                "rule_id": str(item.get("rule_id")) if item.get("rule_id") else None,
                "reason": item.get("reason", "other") if item.get("reason") in valid_reasons else "other",
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result
