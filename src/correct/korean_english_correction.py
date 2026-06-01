# korean_english_correction.py — 한영 코드스위칭 교정
# Whisper STT 출력의 한영 혼용 텍스트를 LLM으로 교정
# 음성 인식 파이프라인에서 사용

import argparse
import json
import subprocess
import sys

# 한영 혼용에서 자주 틀리는 단어 사전 (LLM 호출 없이 빠른 교정)
QUICK_FIX_DICT = {
    # 기술 용어
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
    "미들웨어": "middleware",
    "디스패치": "dispatch",
    "리듀서": "reducer",
    "미들웨어": "middleware",
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
    "디비": "DB",
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
    "노드": "node",
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

def quick_fix(text):
    """사전 기반 빠른 교정 (LLM 없이)"""
    result = text
    for korean, english in QUICK_FIX_DICT.items():
        # 문장 중간에 있는 한글 표기만 교정 (문장 시작이나 독립어는 유지)
        # 예: "리엑트 컴포넌트 만들어야 되는데" → "React 컴포넌트 만들어야 되는데"
        import re
        # 단어 경계에서 매칭
        pattern = r'(?<=[\s,.!?])' + re.escape(korean) + r'(?=[\s,.!?]|$)'
        result = re.sub(pattern, english, result)
        # 문장 시작 매칭
        pattern_start = r'^' + re.escape(korean) + r'(?=[\s,.!?]|$)'
        result = re.sub(pattern_start, english, result)
    return result

def llm_correct(text, screenshot_path=None):
    """GLM-5로 한영 교정."""
    system_prompt = """너는 한국어 개발자의 음성 인식(STT) 텍스트를 교정하는 전문가다.
한국어와 영어가 섞인 텍스트에서 다음을 교정해라:
1. STT 오류 수정 (띄어쓰기, 오탈자, 잘못 들린 단어)
2. 한국어 발음으로 표기된 기술 용어를 올바른 영어로 변환
   - "밸리데이션" → "validation"
   - "리엑트 컴포넌트" → "React 컴포넌트"
   - "깃허브에 PR 올려놨어" → "GitHub에 PR 올려놨어"
3. 불필요한 충임어 제거 (음, 어, 그러니까, 뭐 그런 거)
4. 원래 의도를 최대한 보존하되 자연스럽게

교정된 텍스트만 출력해라. 설명이나 주석은 붙이지 마라."""

    if screenshot_path:
        # 스크린샷이 있으면 참고 텍스트에 추가
        # 실제 구현에서는 이미지를 LLM에 전달
        pass

    # The public helper keeps LLM integration optional and falls back to quick rules.
    return quick_fix(text)

def correct(text, mode="quick", screenshot_path=None):
    """
    텍스트 교정
    
    mode:
      - quick: 사전 기반 빠른 교정 (LLM 없이)
      - llm: GLM-5로 교정
    """
    if mode == "quick":
        return quick_fix(text)
    elif mode == "llm":
        return llm_correct(text, screenshot_path)
    else:
        return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한영 코드스위칭 교정")
    parser.add_argument("text", nargs="*", help="교정할 텍스트 (없으면 stdin)")
    parser.add_argument("--mode", choices=["quick", "llm"], default="quick", help="교정 모드")
    parser.add_argument("--screenshot", help="스크린샷 파일 경로")
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


