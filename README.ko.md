# korean-call-transcriber

한국어 전화 통화 전사 파이프라인 — Whisper + 화자 분리, 자동 TODO/일정/개체명 추출, Obsidian 동기화.

## 주요 기능

- **🎙️ WhisperX 전사** — faster-whisper(CTranslate2) 기반 GPU 가속 한국어 음성 인식
- **👥 화자 분리** — pyannote 기반 2인 화자 식별 + 한국어 존댓말 휴리스틱
- **📝 통합 LLM 추출** — 단일 API 호출로 요약, TODO, 일정, 개체명, 품목, 금액, 리스크, 교정까지 한번에 추출
- **🔧 STT 교정 레이어** — 영구 치환 사전 + 별칭 정규화, 핫리로드 지원
- **📊 갭 분석기** — 파이프라인 건전성 결정론 검사 + 원인 분류
- **🔄 재시도 큐** — JSONL 기반 원자적 재시도 큐, 지수 백오프
- **📓 Obsidian 동기화** — 전사 결과 → 마크다운 자동 변환 + 거래처 인덱싱
- **📧 Gmail 분류기** — 받은편지함 자동 분류 (광고 → 휴지통, 중요 → 하이라이트)
- **📋 이메일 TODO 추출** — 수신 메일에서 LLM으로 실행 항목 추출
- **📅 캘린더 연동** — Google Calendar OAuth2 이벤트 조회
- **💬 SMS 파이프라인** — SMS → 전사 통합 플레이스홀더 모듈
- **📮 네이버 메일 보관** — IMAP 기반 네이버 메일 구조화 JSON 보관
- **✅ 영구 TODO 저장소** — Jaccard 퍼지 중복 제거, 동일 출처 병합, 완료 추적
- **🧠 지식 그래프** — 개체명 관계 추출 및 순회 (거래처 ↔ TODO ↔ 이벤트)
- **📡 시그널 탐지기** — 3밴드 빠른 스코어링 + 아이디어/개체명 자동 추출
- **⚙️ Minions 잡큐** — Postgres 기반 내구성 작업 큐, 팬아웃, DAG, 크래시 복구
- **🔍 상태 검증기** — 상태 파일 존재/노후/무결성 자동 검사

## 빠른 시작

### 준비물

- Python 3.11+
- CUDA 지원 GPU (RTX 3090에서 테스트)
- ffmpeg (PATH에 등록)
- HuggingFace 토큰 (pyannote 접근 권한 필요, 화자 분리용)
- PostgreSQL 16+ (Minions 잡큐용 — 선택사항)

### 설치

```bash
git clone https://github.com/brood-arch/korean-call-transcriber.git
cd korean-call-transcriber
pip install -r requirements.txt
pip install -e .

# 환경 설정 파일 복사 후 편집
cp .env.example .env
# .env에 API 키와 경로 입력
```

### 사용법

#### 1. 오디오 파일 전사

```bash
# 대기 중인 모든 파일 전사
python -m src.transcribe.batch_transcribe

# 단일 파일 전사
python -m src.transcribe.batch_transcribe --file path/to/audio.m4a

# 최신 파일부터 처리
python -m src.transcribe.batch_transcribe --recent-first --limit 10

# 설치형 CLI
kct-transcribe --recent-first --limit 10
```

#### 2. 구조화된 데이터 추출

```bash
# 전체 추출 실행 (요약 + TODO + 개체명 + ...)
python -m src.extract.extract_all

# 설정 검증용 드라이 런
python -m src.extract.extract_all --dry-run

# 오늘 파일만 처리
python -m src.extract.extract_all --today

# 설치형 CLI
kct-extract --today
```

#### 3. 파이프라인 건전성 분석

```bash
# 파이프라인 갭 확인
python -m src.queue.gap_analyzer

# 상세 리포트 생성
python -m src.queue.gap_analyzer --output-json report.json --output-md report.md

# 파이프라인 상태 확인 단축 명령
kct-health
```

#### 4. Obsidian 동기화

```bash
# 새 전사 결과 동기화
python -m src.sync.sync_obsidian

# 드라이 런
python -m src.sync.sync_obsidian --dry-run

# 전체 파일 재동기화
python -m src.sync.sync_obsidian --all

# 설치형 CLI
kct-sync-obsidian --all
```

### 환경 변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `LLM_API_KEY` | LLM API 키 (`ZAI_API_KEY`도 이전 호환용으로 지원) | (필수) |
| `LLM_BASE_URL` | OpenAI 호환 API 베이스 URL (`ZAI_BASE_URL`도 지원) | `https://api.z.ai/api/coding/paas/v4` |
| `LLM_MODEL` | 모델명 | `glm-5.1` |
| `LLM_DISABLE_THINKING` | GLM thinking trace 비활성화 (`auto`, `true`, `false`) | `auto` |
| `AUDIO_DIR` | 오디오 소스 디렉터리 | `data/audio` |
| `TRANSCRIPT_DIR` | 전사 결과 출력 디렉터리 | `output/transcripts` |
| `WHISPER_MODEL` | faster-whisper 모델 | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| `HF_TOKEN_FILE` | HuggingFace 토큰 파일 경로 | (비어있음) |
| `MY_NAME` | 발신자 식별용 이름 | `Me` |
| `GMAIL_ADDRESS` | Gmail IMAP 로그인 주소 | (비어있음) |
| `GMAIL_APP_PASSWORD` | Gmail 앱 비밀번호 | (비어있음) |
| `GCAL_TOKEN_PATH` | Google Calendar OAuth2 토큰 경로 | `state/gcal_token.json` |
| `EMAIL_TODO_STATE` | 이메일 TODO 상태 파일 경로 | `state/email_todo_state.json` |
| `EMAIL_TODO_EXCLUSIONS` | 발신자 제외 목록 경로 | `state/email_todo_exclusions.json` |
| `KCT_STATE_DIR` | 기본 상태 디렉터리 | `state` |
| `MINIONS_DB_HOST` | Minions Postgres 호스트 | `localhost` |
| `MINIONS_DB_PORT` | Minions Postgres 포트 | `5432` |
| `MINIONS_DB_NAME` | Minions 데이터베이스명 | `minions` |
| `MINIONS_DB_USER` | Minions 데이터베이스 사용자 | `minions` |
| `MINIONS_DB_PASS` | Minions 데이터베이스 비밀번호 | (큐 사용 시 필수) |
| `SMS_GATEWAY_URL` | SMS 게이트웨이 API 엔드포인트 | (비어있음) |
| `SMS_API_KEY` | SMS 게이트웨이 API 키 | (비어있음) |
| `NAVER_MAIL_ADDRESS` | 네이버 메일 주소 | (비어있음) |
| `NAVER_MAIL_PASSWORD` | 네이버 메일 비밀번호 또는 앱 비밀번호 | (비어있음) |
| `NAVER_MAIL_HOST` | 네이버 IMAP 호스트 | `imap.naver.com` |
| `NAVER_MAIL_FOLDERS` | 쉼표로 구분된 IMAP 폴더 | `INBOX,"Sent Messages"` |
| `NAVER_MAIL_LIMIT` | 폴더당 최대 메시지 수 | `100` |
| `NAVER_MAIL_STATE_DIR` | 처리된 UID 상태 디렉터리 | `state/naver_mail` |

전체 목록은 [.env.example](.env.example)을 참조하세요.

## 프로젝트 구조

```
src/
├── transcribe/          # WhisperX 전사 엔진
│   ├── batch_transcribe.py   # 배치 전사 메인 스크립트
│   ├── worker.py             # 격리된 서브프로세스 워커
│   └── align_worker.py       # 정렬 + 화자 분리 워커
├── extract/             # LLM 기반 추출
│   ├── extract_all.py        # 통합 추출 (8개 카테고리)
│   ├── extract_entities.py   # 독립 개체명 추출
│   └── extract_schedules.py  # 일정/약속 추출
├── correct/             # STT 교정 레이어
│   └── corrections.py        # 정확한 치환 + 별칭 교정
├── sync/                # 출력 동기화
│   └── sync_obsidian.py      # 전사 → Obsidian 동기화
├── pipeline/            # 공통 유틸리티
│   ├── paths.py              # 중앙 경로 설정
│   ├── utils.py              # 공통 유틸리티
│   ├── health_check.py       # 파이프라인 상태 확인
│   ├── minions_queue.py      # Postgres 기반 내구성 잡큐
│   └── validate_state.py     # 상태 파일 검증
├── integrations/        # 외부 서비스 연동
│   ├── gmail_classifier.py   # Gmail 받은편지함 자동 분류
│   ├── email_todo_extract.py # 이메일 → TODO 추출
│   ├── calendar.py           # Google Calendar 연동
│   ├── sms_handler.py        # SMS 파이프라인 플레이스홀더
│   └── naver_mail.py          # 네이버 메일 IMAP 보관
├── todo/                # TODO 관리
│   └── persistent_store.py   # Jaccard 중복 제거 영구 저장소
├── knowledge/           # 지식 그래프 & 시그널 탐지
│   ├── graph.py              # 개체명 관계 그래프
│   └── signal_detector.py    # 3밴드 빠른 스코어링 + 시그널 탐지
└── queue/               # 파이프라인 상태 & 재시도
    ├── gap_analyzer.py       # 파이프라인 갭 분석
    └── retry_queue.py        # 원자적 재시도 큐
```
