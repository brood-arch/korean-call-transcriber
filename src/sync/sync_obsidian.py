#!/usr/bin/env python3
"""
sync_transcripts_to_obsidian.py
Transcript text files -> Obsidian Vault sync

Usage:
    python -m src.sync.sync_obsidian [--all] [--dry-run]
    python -m src.sync.sync_obsidian --all --dry-run   # 전체 재처리 시뮬레이션

Transcript text files are converted to markdown under an Obsidian-compatible
vault folder, with an optional contact index.
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.config import OBSIDIAN_VAULT, STATE_DIR
from src.config import TRANSCRIPT_DIR as SOURCE_DIR
from src.pipeline.utils import safe_save_json, safe_write_text

log = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────
TRANSCRIPTS_DIR = OBSIDIAN_VAULT / "transcripts"
STATE_FILE = STATE_DIR / "sync_transcripts_state.json"
COUNTERPARTY_DIR = OBSIDIAN_VAULT / "contacts"
COUNTERPARTY_INDEX = COUNTERPARTY_DIR / "contact_index.md"
MIN_CONTENT_LENGTH = 100  # 이 길이 미만이면 스킵

# ── 파일명 파싱 ──────────────────────────────────────────────────
# 패턴1: label_날짜.txt
# 패턴2: 날짜.txt
PATTERN_WITH_NAME = re.compile(r'^(.+?)_(\d{7,12})_(\d{14})\.txt$')
PATTERN_PHONE_ONLY = re.compile(r'^(\d{7,12})_(\d{14})\.txt$')


def parse_filename(filename: str) -> dict | None:
    """파일명에서 거래처명, 전화번호, 날짜를 파싱."""
    m = PATTERN_WITH_NAME.match(filename)
    if m:
        name, phone, datestr = m.group(1), m.group(2), m.group(3)
    else:
        m = PATTERN_PHONE_ONLY.match(filename)
        if m:
            name, phone, datestr = None, m.group(1), m.group(2)
        else:
            return None

    try:
        dt = datetime.strptime(datestr, "%Y%m%d%H%M%S")
    except ValueError:
        return None

    return {
        "counterparty": name,  # None이면 전화번호만 있는 파일
        "phone": phone,
        "datetime": dt,
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S"),
        "datestr": datestr,
    }


def read_source(filepath: Path) -> str:
    """소스 파일 읽기 (UTF-8, CP949 폴백)."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return filepath.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as exc:
            log.debug("Failed to read source file with encoding: %s", exc)
            continue
    # 마지막 수단: 오류 무시
    return filepath.read_text(encoding="utf-8", errors="replace")


def format_phone(phone: str) -> str:
    """숫자로만 된 전화번호를 표시 형식으로 변환."""
    if len(phone) == 11:
        return f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"
    elif len(phone) == 10:
        return f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
    return phone


def generate_md_filename(parsed: dict, existing_count: int) -> str:
    """출력 마크다운 파일명을 생성하며 같은 날짜+거래처에 여러 통화가 있으면 시간 포함."""
    cp = parsed["counterparty"] or parsed["phone"]
    date = parsed["date"]
    if existing_count > 0:
        # 같은 날짜에 이미 파일이 있으면 시간 추가
        time_compact = parsed["datetime"].strftime("%H%M")
        safe_cp = sanitize_filename(cp)
        return f"{date}_{time_compact}_{safe_cp}.md"
    else:
        safe_cp = sanitize_filename(cp)
        return f"{date}_{safe_cp}.md"


def sanitize_filename(name: str) -> str:
    """파일명에 사용할 수 없는 문자를 제거."""
    # Windows 파일명 금지문자 + / 추가
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()


def build_markdown(parsed: dict, content: str, source_path: Path) -> str:
    """전사 텍스트를 Obsidian 호환 마크다운으로 변환."""
    cp = parsed["counterparty"]
    phone = parsed["phone"]
    date = parsed["date"]
    time = parsed["time"]
    filename = source_path.name

    # 거래처 표시명: 이름이 없으면 전화번호
    display_name = cp or format_phone(phone)
    wiki_link = f"[[{cp}]]" if cp else ""

    # 전화번호에 따른 태그 생성
    tags = ["통화전사"]
    if cp:
        # 거래처명에서 핵심 키워드 추출 (첫 단어 또는 괄호 제거)
        tag_base = re.sub(r'[()（）《》]', '', cp).split()[0] if cp else ""
        if tag_base:
            tags.append(tag_base)

    # source_path를 file:/// URI로
    uri_path = str(source_path).replace("\\", "/").replace(" ", "%20")

    lines = [
        "---",
        "type: transcript",
        f"counterparty: \"{cp or ''}\"",
        f"phone: \"{phone}\"",
        f"date: {date}",
        f'time: "{time}"',
        f"source_file: {filename}",
        f"source_path: {str(source_path)}",
        f"tags: [{', '.join(tags)}]",
        "---",
        "",
        f"# 통화 전사 — {display_name} ({date} {parsed['datetime'].strftime('%H:%M')})",
        "",
    ]

    if wiki_link:
        lines.append(f"> **거래처**: {wiki_link}")
    lines.append(f"> **전화**: {format_phone(phone)}")
    lines.append(f"> **원본**: [G드라이브](file:///{uri_path})")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(content)

    return "\n".join(lines)


# ── 상태 관리 ─────────────────────────────────────────────────────
def load_state() -> dict:
    """동기화 처리 상태를 로드."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.debug("Failed to load sync state %s: %s", STATE_FILE, exc)
            return {"processed": {}, "last_run": None}
    return {"processed": {}, "last_run": None}


def save_state(state: dict) -> None:
    """동기화 처리 상태를 저장."""
    state["last_run"] = datetime.now().isoformat()
    safe_save_json(STATE_FILE, state, origin="sync_obsidian")


# ── 거래처 인덱스 업데이트 ────────────────────────────────────────
def update_counterparty_file(counterparty: str, transcript_link: str, date: str) -> None:
    """거래처별 파일에 통화 내역 링크를 추가."""
    if not counterparty:
        return

    safe_name = sanitize_filename(counterparty)
    cp_file = COUNTERPARTY_DIR / f"{safe_name}.md"

    COUNTERPARTY_DIR.mkdir(parents=True, exist_ok=True)

    link_line = f"- [{date}] [[{transcript_link}]]"

    if cp_file.exists():
        content = cp_file.read_text(encoding="utf-8")
        # 이미 있는 링크면 스킵
        if transcript_link in content:
            return

        # ## 통화 기록 섹션 찾기
        section_header = "## 통화 기록"
        if section_header in content:
            # 섹션 헤더 다음에 링크 추가
            parts = content.split(section_header, 1)
            content = parts[0] + section_header + "\n" + link_line + parts[1]
        else:
            # 섹션 없으면 추가
            content += f"\n\n{section_header}\n{link_line}\n"
    else:
        content = (
            f"---\n"
            f"type: counterparty\n"
            f"name: \"{counterparty}\"\n"
            f"---\n\n"
            f"# {counterparty}\n\n"
            f"## 통화 기록\n"
            f"{link_line}\n"
        )

    safe_write_text(cp_file, content)


def update_counterparty_index(new_counterparties: set[str]) -> None:
    """거래처_인덱스.md에 새 거래처가 있으면 추가."""
    if not new_counterparties or not COUNTERPARTY_INDEX.exists():
        return

    content = COUNTERPARTY_INDEX.read_text(encoding="utf-8")

    for cp in sorted(new_counterparties):
        if cp in content:
            continue
        # 마지막 ### 섹션 뒤에 추가
        safe_name = sanitize_filename(cp)
        entry = (
            f"\n### {cp}\n"
            f"- **상태**: 미분류\n"
            f"- **통화 기록**: [[{safe_name}]]\n"
        )
        content += entry

    safe_write_text(COUNTERPARTY_INDEX, content)


# ── 메인 ──────────────────────────────────────────────────────────
def _scan_source_files(source_files, processed, args, key_counts):
    """Scan and parse source files, return parsed list and skip counts."""
    parsed_files = []
    skipped_existing = 0
    skipped_parse = 0
    skipped_short = 0
    for f in source_files:
        if not args.all and f.name in processed:
            skipped_existing += 1
            continue
        parsed = parse_filename(f.name)
        if not parsed:
            skipped_parse += 1
            continue
        try:
            size = f.stat().st_size
        except OSError as exc:
            log.debug("Failed to stat source file %s: %s", f, exc)
            continue
        if size < MIN_CONTENT_LENGTH:
            skipped_short += 1
            continue
        key = (parsed["date"], parsed["counterparty"] or parsed["phone"])
        parsed_files.append((f, parsed, key))
        key_counts[key] += 1
    return parsed_files, skipped_existing, skipped_parse, skipped_short


def _process_parsed_files(parsed_files, key_counts, args):
    """Process parsed files: write markdown, update indexes."""
    key_index = defaultdict(int)
    new_counterparties = set()
    processed_count = 0
    errors = []
    processed_updates = {}
    for src_file, parsed, key in parsed_files:
        try:
            content = read_source(src_file)
            if len(content.strip()) < MIN_CONTENT_LENGTH:
                continue
            has_duplicates = key_counts[key] > 1
            key_index[key] += 1
            cp = parsed["counterparty"] or parsed["phone"]
            safe_cp = sanitize_filename(cp)
            date = parsed["date"]
            if has_duplicates:
                time_compact = parsed["datetime"].strftime("%H%M")
                md_filename = f"{date}_{time_compact}_{safe_cp}.md"
            else:
                md_filename = f"{date}_{safe_cp}.md"
            md_path = TRANSCRIPTS_DIR / md_filename
            md_content = build_markdown(parsed, content, src_file)
            if args.dry_run:
                log.info("  📄 %s → %s", src_file.name, md_filename)
            else:
                safe_write_text(md_path, md_content)
            if parsed["counterparty"]:
                if not args.dry_run:
                    update_counterparty_file(parsed["counterparty"], md_filename, date)
                new_counterparties.add(parsed["counterparty"])
            processed_updates[src_file.name] = {
                "output": md_filename,
                "date": date,
                "counterparty": parsed["counterparty"],
                "processed_at": datetime.now().isoformat(),
            }
            processed_count += 1
        except (ValueError, OSError, RuntimeError) as e:  # noqa: BLE001
            errors.append(f"{src_file.name}: {e}")
    return processed_count, new_counterparties, errors, processed_updates


def _log_summary(args, processed_count, skipped_existing, skipped_parse, skipped_short, new_cps, errors):
    """Print final sync summary."""
    log.info("\n%s", "=" * 50)
    log.info("✅ 처리 완료: %d개", processed_count)
    if skipped_existing:
        log.info("⏭️  이미 처리됨: %d개", skipped_existing)
    if skipped_parse:
        log.info("⚠️  파싱 실패: %d개", skipped_parse)
    if skipped_short:
        log.info("📝 내용 부족(<%d자): %d개", MIN_CONTENT_LENGTH, skipped_short)
    if new_cps:
        log.info("🏢 거래처: %d개 (신규 가능성)", len(new_cps))
    if errors:
        log.error("❌ 오류: %d개", len(errors))
        for e in errors[:10]:
            log.info("   %s", e)
    if args.dry_run:
        log.info("\n💨 dry-run 모드 — 실제 저장 안 함")


def _run_sync(args):
    """Core sync logic."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    state = {} if args.all else load_state()
    processed = state.get("processed", {})
    source_files = sorted(SOURCE_DIR.glob("*.txt"))
    log.info("📂 소스 파일: %d개", len(source_files))

    key_counts = defaultdict(int)
    parsed_files, skipped_existing, skipped_parse, skipped_short = _scan_source_files(
        source_files, processed, args, key_counts,
    )

    processed_count, new_cps, errors, proc_updates = _process_parsed_files(
        parsed_files, key_counts, args,
    )
    processed.update(proc_updates)

    if new_cps and not args.dry_run:
        update_counterparty_index(new_cps)

    if not args.dry_run:
        state["processed"] = processed
        save_state(state)

    _log_summary(args, processed_count, skipped_existing, skipped_parse, skipped_short, new_cps, errors)


def main() -> None:
    """CLI 엔트리포인트: G드라이브 전사본을 Obsidian으로 동기화."""
    parser = argparse.ArgumentParser(
        description="G드라이브 전사본 → Obsidian 동기화"
    )
    parser.add_argument("--all", action="store_true",
                        help="이미 처리한 파일도 재처리")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 저장 없이 결과만 출력")
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        log.info("❌ 소스 디렉토리 없음: %s", SOURCE_DIR)
        sys.exit(1)

    _run_sync(args)


if __name__ == "__main__":
    main()


