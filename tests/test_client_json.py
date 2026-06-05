"""Tests for kct.extract.client — LLM client, JSON extraction, retry policy."""

import json

from kct.extract.client import _extract_json_from_text


class TestExtractJsonFromText:
    def test_plain_json(self):
        text = '{"summary": "test"}'
        assert _extract_json_from_text(text) == {"summary": "test"}

    def test_json_with_prose_prefix(self):
        text = 'Here is the JSON:\n{"summary": "test"}'
        assert _extract_json_from_text(text) == {"summary": "test"}

    def test_json_with_trailing_text(self):
        text = '{"summary": "test"}\n추가 설명'
        assert _extract_json_from_text(text) == {"summary": "test"}

    def test_json_with_markdown_fence(self):
        text = '```json\n{"summary": "test"}\n```'
        assert _extract_json_from_text(text) == {"summary": "test"}

    def test_json_with_plain_fence(self):
        text = '```\n{"summary": "test"}\n```'
        assert _extract_json_from_text(text) == {"summary": "test"}

    def test_empty_string(self):
        assert _extract_json_from_text("") is None

    def test_no_json(self):
        assert _extract_json_from_text("이것은 JSON이 아닙니다") is None

    def test_nested_json(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        text = json.dumps(data, ensure_ascii=False)
        assert _extract_json_from_text(text) == data

    def test_json_with_whitespace(self):
        text = '  \n  {"key": "value"}  \n  '
        assert _extract_json_from_text(text) == {"key": "value"}

    def test_multiple_braces_keeps_outermost(self):
        text = '{"a": {"b": 1}, "c": 2}'
        result = _extract_json_from_text(text)
        assert result == {"a": {"b": 1}, "c": 2}
