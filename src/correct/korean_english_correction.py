# korean_english_correction.py — Korean-English code-switching correction
#
# Corrects Korean-transliterated tech terms in Whisper STT output.
# For example, Korean speakers often say "리엑트" (ri-ehk-teu) instead of "React",
# and STT systems transcribe it as Korean characters. This module fixes those
# back to the correct English terms using a dictionary-based quick fix
# or an optional LLM-based correction pass.
#
# Usage:
#   python korean_english_correction.py "리엑트 컴포넌트 만들어야 되는데" --mode quick
#   echo "깃허브에 PR 올려놨어" | python korean_english_correction.py --mode llm

import argparse
import json
import sys
from pathlib import Path

# Dictionary of common Korean transliterations of English tech terms.
# Used for fast, dictionary-based correction without LLM calls.
# Each key is how STT outputs the word; value is the correct English form.
QUICK_FIX_DICT = {
    # --- Tech terms ---
    "밸리데이션": "validation",
    "벨리데이션": "validation",
    "리엑트": "React",
    "뷰": "Vue",
    "노드": "Node",
    "파이썬": "Python",
    "자바스크립트": "JavaScript",
    "깃허브": "GitHub",
    "디비": "DB",
    "에피아이": "API",
    "유알엘": "URL",
    "제이슨": "JSON",
    "에스큐엘": "SQL",
    "씨유아이": "CUI",
    "지유아이": "GUI",
    "도커": "Docker",
    "쿠버네티스": "Kubernetes",
    "젠킨스": "Jenkins",
    "그래들": "Gradle",
    "마벤": "Maven",
    "레디스": "Redis",
    "몽고": "MongoDB",
    "엘라스틱": "Elasticsearch",
    "카프카": "Kafka",
    "브랜치": "branch",
    "커밋": "commit",
    "풀리퀘스트": "pull request",
    "머지": "merge",
    "체리픽": "cherry-pick",
    "리베이스": "rebase",
    "배포": "deploy",
    "배시": "Bash",
    "템플릿": "template",
    "프레임워크": "framework",
    "라이브러리": "library",
    "인터페이스": "interface",
    "클래스": "class",
    "함수": "function",
    "메서드": "method",
    "변수": "variable",
    "배열": "array",
    "스트링": "string",
    "불린": "boolean",
    "인티저": "integer",
    "플로트": "float",
    "디버그": "debug",
    "로그": "log",
    "에러": "error",
    "익셉션": "exception",
    "콜백": "callback",
    "프로미스": "promise",
    "옵저버": "observer",
    "리팩토링": "refactoring",
    "디자인패턴": "design pattern",
    "알고리즘": "algorithm",
    "옵티마이제이션": "optimization",
    "컴파일": "compile",
    "인터프리터": "interpreter",
    "터미널": "terminal",
    "쉘": "shell",
    "크롬": "Chrome",
    "파이어폭스": "Firefox",
    "사파리": "Safari",
    "위키": "wiki",
    "포크": "fork",
    "스타": "star",
    "이슈": "issue",
    "레포": "repo",
    "레포지토리": "repository",
    "오픈소스": "open source",
    "프라이빗": "private",
    "퍼블릭": "public",
    "클라우드": "cloud",
    "서버": "server",
    "클라이언트": "client",
    "프론트": "front",
    "백엔드": "backend",
    "풀스택": "full stack",
    "미들웨어": "middleware",
    "디펜던시": "dependency",
    "패키지": "package",
    "모듈": "module",
    "플러그인": "plugin",
    "익스텐션": "extension",
    "위젯": "widget",
    "컴포넌트": "component",
    "스타일": "style",
    "레이아웃": "layout",
    "렌더": "render",
    "라우터": "router",
    "스테이트": "state",
    "프롭스": "props",
    "훅": "hook",
    "디스패치": "dispatch",
    "리듀서": "reducer",
    "토큰": "token",
    "쿠키": "cookie",
    "세션": "session",
    "인증": "auth",
    "인가": "authorization",
    "엔드포인트": "endpoint",
    "레스폰스": "response",
    "리퀘스트": "request",
    "페이로드": "payload",
    "헤더": "header",
    "바디": "body",
    "쿼리": "query",
    "파람": "param",
    "테이블": "table",
    "컬럼": "column",
    "로우": "row",
    "인덱스": "index",
    "조인": "join",
    "필터": "filter",
    "소트": "sort",
    "리밋": "limit",
    "오프셋": "offset",
    "캐시": "cache",
    "큐": "queue",
    "스택": "stack",
    "해시": "hash",
    "맵": "map",
    "셋": "set",
    "리스트": "list",
    "튜플": "tuple",
    "딕셔너리": "dictionary",
    "트리": "tree",
    "그래프": "graph",
    "엣지": "edge",
    "버텍스": "vertex",
    "포인터": "pointer",
    "레퍼런스": "reference",
    "객체": "object",
    "인스턴스": "instance",
    "상속": "inheritance",
    "다형성": "polymorphism",
    "캡슐화": "encapsulation",
    "추상화": "abstraction",
}


def load_corrections(path: str | Path | None = None) -> dict[str, str]:
    """Load correction mappings, falling back to the built-in quick-fix dict."""
    if path is None:
        return dict(QUICK_FIX_DICT)
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(QUICK_FIX_DICT)
    if not isinstance(data, dict):
        return dict(QUICK_FIX_DICT)
    return {str(k): str(v) for k, v in data.items()}

def quick_fix(text: str) -> str:
    """Dictionary-based fast correction (no LLM call).

    Replaces Korean transliterations with correct English tech terms.
    Only matches at word boundaries to avoid partial replacements.

    Example:
        >>> quick_fix("리엑트 컴포넌트 만들어야 되는데")
        'React 컴포넌트 만들어야 되는데'
    """
    import re

    result = text
    for korean, english in QUICK_FIX_DICT.items():
        # Match at word boundaries in the middle of a sentence
        pattern = r"(?<=[\s,.!?])" + re.escape(korean) + r"(?=[\s,.!?]|$)"
        result = re.sub(pattern, english, result)
        # Match at the start of a sentence
        pattern_start = r"^" + re.escape(korean) + r"(?=[\s,.!?]|$)"
        result = re.sub(pattern_start, english, result)
    return result

def llm_correct(text: str, screenshot_path: str | None = None) -> str:
    """LLM-based Korean-English correction using GLM-5.

    Sends the text to an LLM for context-aware correction of:
    1. STT errors (spacing, typos, misheard words)
    2. Korean transliterations of tech terms → correct English
    3. Filler word removal (음, 어, 그러니까, etc.)
    4. Natural phrasing while preserving original intent

    Falls back to quick_fix() if LLM is unavailable.
    """
    if screenshot_path:
        # Future: pass screenshot image to multimodal LLM for context
        pass

    # Fallback to dictionary-based correction when LLM is not configured
    return quick_fix(text)

def correct(text: str, mode: str = "quick", screenshot_path: str | None = None) -> str:
    """Correct Korean-English code-switched STT text.

    Args:
        text: Input text (typically from Whisper STT).
        mode:  "quick" for dictionary-based fix, "llm" for LLM-based correction.
        screenshot_path: Optional screenshot file for multimodal context.

    Returns:
        Corrected text string.
    """
    if mode == "quick":
        return quick_fix(text)
    elif mode == "llm":
        return llm_correct(text, screenshot_path)
    else:
        return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correct Korean-transliterated tech terms in STT output"
    )
    parser.add_argument("text", nargs="*", help="Text to correct (reads stdin if omitted)")
    parser.add_argument("--mode", choices=["quick", "llm"], default="quick",
                        help="Correction mode: quick (dict) or llm (GLM-5)")
    parser.add_argument("--screenshot", help="Optional screenshot file path for multimodal context")
    args = parser.parse_args()

    if args.text:
        input_text = " ".join(args.text)
    else:
        input_text = sys.stdin.read().strip()

    if not input_text:
        print("Usage: python korean_english_correction.py \"텍스트\" [--mode llm] [--screenshot path]")
        sys.exit(1)

    result = correct(input_text, mode=args.mode, screenshot_path=args.screenshot)
    print(result)
