#!/usr/bin/env python3
"""Deterministic transcription pipeline gap analyzer.

Builds a repeatable cause taxonomy for the transcription
pipeline without mutating pipeline state.  It intentionally treats
``{audio_stem}_{HHMMSS}.txt`` files as derivative diarization/recheck artifacts,
not as canonical transcripts or TODO-extraction inputs.

Exit code policy:
  0 = normal/no actionable gaps
  1 = warning/post-processing or intentionally excluded gaps exist
  2 = danger/missing canonical transcript or non-blacklisted transcription failure
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))

try:
    from src.pipeline.paths import AUDIO_DIR, TRANSCRIPT_DIR, STATE_DIR, LOG_DIR, WORKSPACE
except Exception:
    WORKSPACE = Path(os.environ.get("KCT_WORKSPACE", Path.cwd()))
    AUDIO_DIR = Path(os.environ.get("KCT_AUDIO_DIR", WORKSPACE / "data" / "audio"))
    TRANSCRIPT_DIR = Path(os.environ.get("KCT_TRANSCRIPT_DIR", WORKSPACE / "output" / "transcripts"))
    STATE_DIR = Path(os.environ.get("KCT_STATE_DIR", WORKSPACE / "state"))
    LOG_DIR = Path(os.environ.get("KCT_LOG_DIR", WORKSPACE / "logs"))

DEFAULT_AUDIO_DIR = AUDIO_DIR
DEFAULT_TRANSCRIPT_DIR = TRANSCRIPT_DIR
DEFAULT_STATE_DIR = STATE_DIR
DEFAULT_LOG_DIR = LOG_DIR

BLACKLIST_FILE = DEFAULT_STATE_DIR / "transcribe_blacklist.json"
CHROMA_STATE_FILE = DEFAULT_STATE_DIR / "chroma_index_state.json"
OBSIDIAN_STATE_FILE = DEFAULT_STATE_DIR / "sync_transcripts_state.json"
INTEGRATED_EXTRACTION_DIR = DEFAULT_STATE_DIR / "integrated_extraction"
TRANSCRIBE_LOG = DEFAULT_LOG_DIR / "transcribe_vv.log"

CAUSE_ORDER = [
    "missing_sync",
    "missing_transcript",
    "blacklisted",
    "transcription_failed",
    "diarization_failed",
    "entity_pending",
    "rag_pending",
    "obsidian_pending",
    "derived_excluded",
]

NEXT_ACTION = {
    "missing_sync": "원본 오디오 경로/동기화 상태 확인",
    "missing_transcript": "batch_transcribe_whisperx.py --file 로 즉시 전사 재처리",
    "blacklisted": "전사 제외 의도가 맞으면 유지; 업무 통화로 판단되면 blacklist에서 제거 후 재처리",
    "transcription_failed": "실패 원인 확인 후 재시도 또는 blacklist 정책 결정",
    "diarization_failed": "파생 재점검본이 있으면 원본 대체 검토; 없으면 run_diarization_recheck_batch.py 대상에 추가",
    "entity_pending": "src.extract.extract_all 또는 integrated extraction 배치 재개",
    "rag_pending": "검색 인덱스 incremental 실행",
    "obsidian_pending": "sync_transcripts_to_obsidian.py 또는 export 단계 재실행",
    "derived_excluded": "재점검 파생본이므로 TODO/전사 gap에서는 제외",
}

PRIORITY = {
    "missing_sync": "P0",
    "missing_transcript": "P0",
    "transcription_failed": "P1",
    "diarization_failed": "P2",
    "rag_pending": "P2",
    "obsidian_pending": "P2",
    "entity_pending": "P2",
    "blacklisted": "P3",
    "derived_excluded": "P4",
}


class AnalyzerError(RuntimeError):
    pass


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text("utf-8-sig"))


def canonical_transcript_stem(stem: str) -> str:
    """Return source audio stem for timestamp-suffixed transcript variants."""
    m = re.match(r"^(.+_\d{14})_\d{6}$", stem)
    return m.group(1) if m else stem


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=KST).isoformat()


def file_record(path: Path, reason: str, *, stem: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "file": path.name,
        "stem": stem or path.stem,
        "reason": reason,
        "priority": PRIORITY[reason],
        "next_action": NEXT_ACTION[reason],
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime": iso_mtime(path) if path.exists() else None,
    }
    if extra:
        data.update(extra)
    return data


def load_blacklist(path: Path) -> tuple[dict[str, dict[str, Any]], set[str], set[str]]:
    raw = load_json(path, {})
    entries = {k: v for k, v in raw.items() if k != "_meta" and isinstance(v, dict)}
    blacklisted = {k for k, v in entries.items() if v.get("blacklisted_at")}
    failed = {
        k
        for k, v in entries.items()
        if not v.get("blacklisted_at") and int(v.get("failures") or 0) > 0
    }
    return entries, blacklisted, failed


def load_chroma_basenames(path: Path) -> set[str]:
    raw = load_json(path, {})
    files = raw.get("files", {}) if isinstance(raw, dict) else {}
    return {Path(str(k).replace("\\", "/")).name for k in files}


def load_obsidian_processed(path: Path) -> set[str]:
    raw = load_json(path, {})
    processed = raw.get("processed", {}) if isinstance(raw, dict) else {}
    return {str(k) for k in processed}


def load_integrated_processed(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    for batch in sorted(path.glob("batch_*.json")):
        raw = load_json(batch, {})
        for stem in raw.get("files", []) if isinstance(raw, dict) else []:
            processed.add(str(stem))
    return processed


def load_diarization_failed(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    failed: set[str] = set()
    pattern = re.compile(r"Speaker diarization JSON parse failed for (.+?)\.m4a:")
    for line in log_path.read_text("utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            failed.add(m.group(1))
    return failed


def analyze(
    *,
    workspace: Path,
    audio_dir: Path,
    transcript_dir: Path,
    blacklist_file: Path,
    chroma_state_file: Path,
    obsidian_state_file: Path,
    integrated_extraction_dir: Path,
    transcribe_log: Path,
) -> dict[str, Any]:
    if not audio_dir.exists():
        raise AnalyzerError(f"audio_dir not found: {audio_dir}")
    if not transcript_dir.exists():
        raise AnalyzerError(f"transcript_dir not found: {transcript_dir}")

    audio_files = sorted(
        [p for p in audio_dir.glob("*.m4a") if p.is_file() and p.stat().st_size > 1024],
        key=lambda p: p.name,
    )
    transcript_files = sorted([p for p in transcript_dir.glob("*.txt") if p.is_file()], key=lambda p: p.name)
    audio_by_stem = {p.stem: p for p in audio_files}
    transcript_by_stem = {p.stem: p for p in transcript_files}

    blacklist_entries, blacklisted_stems, failed_stems = load_blacklist(blacklist_file)
    chroma_basenames = load_chroma_basenames(chroma_state_file)
    obsidian_processed = load_obsidian_processed(obsidian_state_file)
    integrated_processed = load_integrated_processed(integrated_extraction_dir)
    diarization_failed = load_diarization_failed(transcribe_log)

    cause_files: dict[str, list[dict[str, Any]]] = {k: [] for k in CAUSE_ORDER}

    # Canonical transcription gap taxonomy: audio stem -> stem.txt.
    for stem, audio_path in sorted(audio_by_stem.items()):
        if stem in transcript_by_stem:
            continue
        if stem in blacklisted_stems:
            cause_files["blacklisted"].append(
                file_record(audio_path, "blacklisted", stem=stem, extra={"blacklist": blacklist_entries.get(stem, {})})
            )
        elif stem in failed_stems:
            cause_files["transcription_failed"].append(
                file_record(audio_path, "transcription_failed", stem=stem, extra={"blacklist": blacklist_entries.get(stem, {})})
            )
        else:
            cause_files["missing_transcript"].append(file_record(audio_path, "missing_transcript", stem=stem))

    # Files in transcript dir that are derivative timestamp-suffixed rechecks.
    for txt in transcript_files:
        canonical = canonical_transcript_stem(txt.stem)
        if canonical != txt.stem and canonical in audio_by_stem:
            cause_files["derived_excluded"].append(
                file_record(txt, "derived_excluded", stem=txt.stem, extra={"canonical_stem": canonical})
            )

    # Downstream processing gaps are transcript-file oriented.
    for txt in transcript_files:
        if txt.name not in chroma_basenames:
            cause_files["rag_pending"].append(file_record(txt, "rag_pending", stem=txt.stem))
        if txt.name not in obsidian_processed:
            cause_files["obsidian_pending"].append(file_record(txt, "obsidian_pending", stem=txt.stem))
        if txt.stem not in integrated_processed:
            cause_files["entity_pending"].append(file_record(txt, "entity_pending", stem=txt.stem))

    for stem in sorted(diarization_failed):
        path = audio_by_stem.get(stem) or audio_dir / f"{stem}.m4a"
        cause_files["diarization_failed"].append(file_record(path, "diarization_failed", stem=stem))

    category_counts = {k: len(cause_files[k]) for k in CAUSE_ORDER}
    exact_complete = sum(1 for stem in audio_by_stem if stem in transcript_by_stem)

    counts = {
        "source_audio_total_valid_m4a": len(audio_files),
        "transcript_txt_total_all": len(transcript_files),
        "pipeline_status_style_complete_txt_count": len(transcript_files),
        "pipeline_status_style_gap_audio_minus_txt": len(audio_files) - len(transcript_files),
        "exact_transcript_complete_count": exact_complete,
        "exact_transcript_gap_count": len(audio_files) - exact_complete,
        "derived_transcripts_excluded_count": category_counts["derived_excluded"],
        "blacklist_entries_total": len(blacklist_entries),
        "chroma_index_state_files": len(chroma_basenames),
        "rag_pending_count": category_counts["rag_pending"],
        "obsidian_processed_state_count": len(obsidian_processed),
        "obsidian_pending_count": category_counts["obsidian_pending"],
        "integrated_entity_processed_stems_from_batches": len(integrated_processed),
        "entity_pending_count": category_counts["entity_pending"],
        "diarization_failed_log_unique_count": category_counts["diarization_failed"],
        "reprocess_queue_candidates_count": (
            category_counts["missing_transcript"]
            + category_counts["transcription_failed"]
            + category_counts["diarization_failed"]
        ),
    }

    if category_counts["missing_transcript"] or category_counts["transcription_failed"] or category_counts["missing_sync"]:
        health = "danger"
        exit_code = 2
    elif any(category_counts[k] for k in CAUSE_ORDER if k != "derived_excluded"):
        health = "warning"
        exit_code = 1
    else:
        health = "normal"
        exit_code = 0

    action_queue = []
    for reason in CAUSE_ORDER:
        if reason in {"blacklisted", "derived_excluded", "entity_pending", "obsidian_pending"}:
            # Keep very large/background queues out of the immediate queue.
            continue
        action_queue.extend(cause_files[reason])
    action_queue = sorted(action_queue, key=lambda r: (r["priority"], r["reason"], r["file"]))

    return {
        "schema_version": 1,
        "generated_at": "omitted_for_deterministic_reruns",
        "workspace": str(workspace),
        "source_dir": str(audio_dir),
        "transcript_dir": str(transcript_dir),
        "health": health,
        "exit_code": exit_code,
        "method": {
            "pipeline_style_count": "Count all *.txt under transcript_dir, including derived recheck txt files.",
            "exact_transcript_count": "Require transcript_dir/{audio_stem}.txt; derived {audio_stem}_HHMMSS.txt is excluded from canonical completion.",
            "blacklist": "memory/state/transcribe_blacklist.json entries with blacklisted_at are intentionally excluded/held.",
            "rag": "memory/state/chroma_index_state.json file keys compared by basename.",
            "obsidian": "memory/state/sync_transcripts_state.json processed keys compared by transcript filename.",
            "entity": "memory/state/integrated_extraction/batch_*.json files compared by stem; checkpoint.json absence is tolerated.",
            "diarization": "logs/transcribe_vv.log unique 'Speaker diarization JSON parse failed for *.m4a' stems.",
        },
        "counts": counts,
        "category_counts": category_counts,
        "cause_files": cause_files,
        "action_queue": action_queue,
    }


def render_markdown(report: dict[str, Any], json_path: Path | None = None, max_action_rows: int = 30) -> str:
    counts = report["counts"]
    category_counts = report["category_counts"]
    lines = [
        "# 전사 파이프라인 Gap Analyzer Report",
        "",
        f"- health: {report['health']} (exit_code={report['exit_code']})",
        f"- 오디오 원본: `{report['source_dir']}`",
        f"- 전사본: `{report['transcript_dir']}`",
    ]
    if json_path:
        lines.append(f"- JSON: `{json_path}`")
    lines += [
        "",
        "## 결론",
        "",
        f"- live pipeline-style count: {counts['pipeline_status_style_complete_txt_count']}/{counts['source_audio_total_valid_m4a']} (gap {counts['pipeline_status_style_gap_audio_minus_txt']}) — 단순 *.txt 카운트 방식.",
        f"- 실제 재전사 pending 기준(exact stem.txt): {counts['exact_transcript_complete_count']}/{counts['source_audio_total_valid_m4a']} (gap {counts['exact_transcript_gap_count']}).",
        f"- exact gap 중 blacklist {category_counts['blacklisted']}건, missing_transcript {category_counts['missing_transcript']}건, transcription_failed {category_counts['transcription_failed']}건.",
        f"- 재처리 큐 후보: {counts['reprocess_queue_candidates_count']}건 (blacklist/derived 제외).",
        "",
        "## 카테고리별 카운트",
        "",
        "| category | count | next_action |",
        "|---|---:|---|",
    ]
    for reason in CAUSE_ORDER:
        lines.append(f"| {reason} | {category_counts[reason]} | {NEXT_ACTION[reason]} |")

    lines += ["", f"## 즉시 액션 큐 상위 {max_action_rows}건", "", "| priority | reason | file | next_action |", "|---|---|---|---|"]
    for row in report["action_queue"][:max_action_rows]:
        lines.append(f"| {row['priority']} | {row['reason']} | `{row['file']}` | {row['next_action']} |")
    if not report["action_queue"]:
        lines.append("| - | - | - | 즉시 액션 없음 |")

    lines += [
        "",
        "## 재현 방법",
        "",
        "- total: `통화녹음/*.m4a` 중 size > 1024 bytes",
        "- pipeline complete: `전사본/*.txt` 전체 수",
        "- exact complete: `전사본/{audio_stem}.txt` 존재 수",
        "- blacklist: `memory/state/transcribe_blacklist.json`의 `blacklisted_at`",
        "- RAG: `memory/state/chroma_index_state.json.files` basename 비교",
        "- Obsidian: `memory/state/sync_transcripts_state.json.processed` key 비교",
        "- Entity: `memory/state/integrated_extraction/batch_*.json.files` stem 비교",
        "- Derived: `{audio_stem}_{HHMMSS}.txt`는 파생본으로 TODO/전사 gap에서 제외",
        "",
    ]
    return "\n".join(lines)


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze transcription pipeline gaps by cause taxonomy")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--blacklist-file", type=Path, default=BLACKLIST_FILE)
    parser.add_argument("--chroma-state-file", type=Path, default=CHROMA_STATE_FILE)
    parser.add_argument("--obsidian-state-file", type=Path, default=OBSIDIAN_STATE_FILE)
    parser.add_argument("--integrated-extraction-dir", type=Path, default=INTEGRATED_EXTRACTION_DIR)
    parser.add_argument("--transcribe-log", type=Path, default=TRANSCRIBE_LOG)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true", help="Print full JSON report to stdout")
    parser.add_argument("--no-exit-status", action="store_true", help="Always exit 0 after writing report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        report = analyze(
            workspace=args.workspace,
            audio_dir=args.audio_dir,
            transcript_dir=args.transcript_dir,
            blacklist_file=args.blacklist_file,
            chroma_state_file=args.chroma_state_file,
            obsidian_state_file=args.obsidian_state_file,
            integrated_extraction_dir=args.integrated_extraction_dir,
            transcribe_log=args.transcribe_log,
        )
    except AnalyzerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        safe_write_text(args.output_json, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        safe_write_text(args.output_md, render_markdown(report, args.output_json))

    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        cc = report["category_counts"]
        c = report["counts"]
        print(
            f"health={report['health']} exit_code={report['exit_code']} "
            f"exact={c['exact_transcript_complete_count']}/{c['source_audio_total_valid_m4a']} "
            f"gap={c['exact_transcript_gap_count']} "
            f"missing={cc['missing_transcript']} blacklisted={cc['blacklisted']} "
            f"diarization_failed={cc['diarization_failed']} rag_pending={cc['rag_pending']} "
            f"obsidian_pending={cc['obsidian_pending']} entity_pending={cc['entity_pending']} "
            f"derived_excluded={cc['derived_excluded']}"
        )
    return 0 if args.no_exit_status else int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

