import json

from kct.correct.korean_english_correction import correct, load_corrections, quick_fix


def test_quick_fix_replaces_term_at_start():
    assert quick_fix("리엑트 컴포넌트") == "React component"


def test_quick_fix_replaces_term_after_space():
    assert quick_fix("오늘 깃허브 확인") == "오늘 GitHub 확인"


def test_quick_fix_does_not_replace_inside_word():
    assert quick_fix("자리엑트") == "자리엑트"


def test_correct_llm_falls_back_to_quick_fix():
    assert correct("파이썬 코드", mode="llm") == "Python 코드"


def test_correct_unknown_mode_returns_original():
    assert correct("리엑트", mode="unknown") == "리엑트"


def test_load_corrections_default_contains_builtin():
    data = load_corrections()
    assert data["리엑트"] == "React"


def test_load_corrections_custom_file(tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"테스트": "test"}, ensure_ascii=False), encoding="utf-8")
    assert load_corrections(path) == {"테스트": "test"}


def test_load_corrections_bad_file_falls_back(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")
    assert load_corrections(path)["깃허브"] == "GitHub"
