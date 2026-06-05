from src.extract.parser import parse_unified_response


def test_parse_valid_unified_json():
    result = parse_unified_response('{"summary":{"one_line":"ok"},"todos":[{"title":"Call back","owner":"me"}]}')
    assert result["parse_error"] is False
    assert result["summary"]["one_line"] == "ok"
    assert result["todos"][0]["owner"] == "me"


def test_parse_strips_markdown_fence():
    result = parse_unified_response('```json\n{"entities":[{"name":"Acme","type":"Organization"}]}\n```')
    assert result["entities"][0]["name"] == "Acme"


def test_parse_extracts_json_from_mixed_text():
    result = parse_unified_response('prefix {"money":[{"amount":"12000","kind":"price"}]} suffix')
    assert result["money"][0]["amount"] == 12000


def test_invalid_json_returns_parse_error():
    result = parse_unified_response("not json")
    assert result["parse_error"] is True
    assert result["raw"] == "not json"


def test_todo_validation_normalizes_bad_enums():
    result = parse_unified_response(
        '{"todos":[{"title":"Do it",'
        '"owner":"other","priority":"urgent",'
        '"status":"weird"}]}'
    )
    todo = result["todos"][0]
    assert todo["owner"] == "unknown"
    assert todo["priority"] == "medium"
    assert todo["status"] == "new"


def test_entity_unknown_type_becomes_other():
    result = parse_unified_response('{"entities":[{"name":"Thing","type":"BadType"}]}')
    assert result["entities"][0]["type"] == "Other"


def test_money_bad_amount_defaults_zero():
    result = parse_unified_response('{"money":[{"amount":"not-a-number"}]}')
    assert result["money"][0]["amount"] == 0


def test_non_dict_summary_is_empty():
    result = parse_unified_response('{"summary":["bad"]}')
    assert result["summary"] == {}
