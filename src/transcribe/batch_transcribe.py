"""Batch transcription with WhisperX.

Strategy: faster-whisper (main process) + align/diarize (subprocess to avoid DLL crash).

The ctranslate2 DLL used by faster-whisper conflicts with whisperx's pyannote imports
when loaded in the same process. Solution: run transcription in main process, then
spawn a child for alignment and diarization.

Usage:
    python batch_transcribe.py                    # Process all pending files
    python batch_transcribe.py --file audio.m4a   # Transcribe one file
    python batch_transcribe.py --recent-first      # Process newest files first
    python batch_transcribe.py --dry-run           # Preview without processing

Configuration via environment variables:
    AUDIO_DIR           — Source audio directory (default: data/audio)
    TRANSCRIPT_DIR      — Output transcript directory (default: data/transcripts)
    WHISPER_MODEL       — faster-whisper model path or HuggingFace ID
    HF_TOKEN_FILE       — Path to HuggingFace token file for pyannote
    MY_NAME             — Speaker name for caller identification (default: "Me")
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import psutil

from src.config import (
    AUDIO_DIR,
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_PARTIAL,
    LOG_DIR,
    STATE_DIR,
    TRANSCRIPT_DIR,
)
from src.config import (
    HF_TOKEN_FILE as _HF_TOKEN_FILE,
)
from src.config import (
    MY_NAME as _MY_NAME,
)
from src.config import (
    TRANSCRIBE_LOG as _TRANSCRIBE_LOG,
)
from src.config import (
    WHISPER_COMPUTE_TYPE as _COMPUTE_TYPE,
)
from src.config import (
    WHISPER_MODEL as _WHISPER_MODEL,
)
from src.correct.transcription_corrections import apply_corrections, ensure_rules_file
from src.pipeline.utils import safe_write_text

# ffmpeg setup: ensure the Python Scripts dir (which may contain ffmpeg.exe) is in PATH
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

# ── Configuration ──────────────────────────────────────────────────────
SOURCE_DIR = AUDIO_DIR
OUTPUT_DIR = TRANSCRIPT_DIR
LOG_FILE = _TRANSCRIBE_LOG
BLACKLIST_FILE = STATE_DIR / "transcribe_blacklist.json"
HF_TOKEN_FILE = Path(_HF_TOKEN_FILE) if _HF_TOKEN_FILE else Path()
MY_NAME = _MY_NAME

WHISPER_MODEL = os.environ.get(
    "WHISPER_MODEL",
    _WHISPER_MODEL or "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
)
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", _COMPUTE_TYPE or "float16")
LANGUAGE = "ko"
MAX_AUDIO_DURATION_SEC = 3600  # 60 minutes
LONG_AUDIO_CHUNK_THRESHOLD_SEC = 300
LONG_AUDIO_CHUNK_SEC = 300
LONG_AUDIO_FAST_PATH_SEC = 30 * 60  # Skip align/diarize on very long calls

QUALITY_LOG = LOG_DIR / "transcribe_quality.jsonl"
CORRECTION_STATS = STATE_DIR / "correction_stats.json"

GPU_MIN_FREE_MB = 4000
GPU_MIN_FREE_PER_FILE = 2000
RAM_THRESHOLD_PCT = 85
MAX_CONSECUTIVE_FAILURES = 3

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE), level=logging.INFO, encoding="utf-8",
    format="%(asctime)s %(message)s", force=True,
)
log = logging.info


def parse_args():
    p = argparse.ArgumentParser(description="Batch transcription with WhisperX")
    p.add_argument("--limit", type=int, default=0, help="Max files to process")
    p.add_argument("--file", help="Transcribe exactly one audio file")
    p.add_argument("--recent-first", action="store_true", help="Process newest files first")
    p.add_argument("--force", action="store_true", help="Overwrite existing transcripts")
    p.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    p.add_argument("--verbose", action="store_true", help="Print extra diagnostic output")
    p.add_argument("--json", action="store_true", help="Print machine-readable summary JSON")
    args = p.parse_args()
    if not args.file and os.environ.get("TRANSCRIBE_FILE"):
        args.file = os.environ["TRANSCRIBE_FILE"]
    if os.environ.get("TRANSCRIBE_FORCE"):
        args.force = True
    return args


# ── GPU helpers ──────────────────────────────────────────────────────────

def get_gpu_free_mb() -> int:
    """Query free GPU memory in MB via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        return int((r.stdout or "0").strip().splitlines()[0])
    except Exception as exc:
        logging.debug("GPU memory query failed: %s", exc)
        return 0


def resolve_whisper_model() -> str:
    """Return the first usable faster-whisper model path/name."""
    candidates = [WHISPER_MODEL]
    # Add fallback candidates
    fallbacks = ["mobiuslabsgmbh/faster-whisper-large-v3-turbo"]
    for fb in fallbacks:
        if fb not in candidates:
            candidates.append(fb)
    for candidate in candidates:
        if ":\\" in candidate or candidate.startswith("/"):
            if Path(candidate).exists():
                return candidate
        else:
            return candidate
    return WHISPER_MODEL


def kill_gpu_hogs() -> list[str]:
    """Kill Python processes holding GPU memory (emergency recovery)."""
    killed = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        for line in (r.stdout or "").strip().splitlines():
            parts = [p.strip() for p in line.split(", ")]
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            name = parts[-1] if len(parts) >= 2 else ""
            if "python" in name.lower():
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)
                killed.append(f"{name}(PID {pid})")
    except Exception as exc:
        logging.debug("Failed to kill GPU processes: %s", exc)
    return killed


def get_audio_duration(file) -> float:
    """Get audio duration in seconds via ffmpeg."""
    import re
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", str(file), "-f", "null", "-"],
            capture_output=True, text=True, check=False, timeout=30,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception as exc:
        logging.debug("ffmpeg duration probe failed for %s: %s", file, exc)
    try:
        return Path(file).stat().st_size / 1024 / 1024 * 60
    except Exception as exc:
        logging.debug("Stat-based duration fallback failed for %s: %s", file, exc)
        return 0


# ── Blacklist management ─────────────────────────────────────────────────

def _load_blacklist() -> dict:
    if BLACKLIST_FILE.exists():
        try:
            return json.loads(BLACKLIST_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.debug("Failed to load blacklist %s: %s", BLACKLIST_FILE, exc)
            return {}
    return {}


def _save_blacklist(data: dict):
    BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BLACKLIST_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(BLACKLIST_FILE)


def blacklist_add(stem: str, error_msg: str, count: bool = True) -> int:
    """Add or update a blacklist entry. Returns failure count."""
    data = _load_blacklist()
    if stem not in data:
        data[stem] = {"failures": 0, "last_error": "", "blacklisted_at": None}
    entry = data[stem]
    if count:
        entry["failures"] += 1
    entry["last_error"] = error_msg[:500]
    entry["last_attempt"] = datetime.now().isoformat()
    if entry["failures"] >= MAX_CONSECUTIVE_FAILURES and not entry.get("blacklisted_at"):
        entry["blacklisted_at"] = datetime.now().isoformat()
    _save_blacklist(data)
    return entry["failures"]


def is_blacklisted(stem: str) -> bool:
    data = _load_blacklist()
    return stem in data and data[stem].get("blacklisted_at") is not None


def get_pending(recent_first: bool = False) -> list[Path]:
    """Return list of audio files that need transcription."""
    if not SOURCE_DIR.exists():
        return []
    audio_files = [f for f in SOURCE_DIR.glob("*.m4a") if f.stat().st_size > 1024]
    pending = [
        f for f in audio_files
        if not is_blacklisted(f.stem) and not (OUTPUT_DIR / f"{f.stem}.txt").exists()
    ]
    if recent_first:
        pending.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    else:
        pending.sort(key=lambda f: f.name)
    return pending


# ── Speaker name mapping ────────────────────────────────────────────────

def parse_caller_info(stem: str) -> tuple[str | None, str | None]:
    """Extract caller name and phone from file stem.

    Format: {label}_{datetime}, for example CallerName_20260507164244
    Returns (caller_name, phone) or (None, None).
    """
    import re
    m = re.match(r"^(.+?)_(\d{10,11})_\d{14}$", stem)
    if m:
        return m.group(1), m.group(2)
    return None, None


def map_speakers(segments: list[dict], caller_name: str, caller_phone: str) -> list[dict]:
    """Replace SPEAKER_00/01 with actual names using speech analysis heuristics.

    Strategy: Determine which SPEAKER_ID is the caller vs the user using:
    1. Speech duration ratio (caller often speaks longer in business calls)
    2. Greeting patterns (first greeter = caller in outgoing calls)
    3. Honorific patterns (counterpart uses honorifics toward the user)
    """
    if not caller_name or not any("SPEAKER" in str(s.get("speaker", "")).upper() for s in segments):
        return segments

    speakers = set(seg.get("speaker") for seg in segments if seg.get("speaker"))
    if len(speakers) != 2:
        return segments

    sp_list = sorted(speakers)
    sp0, sp1 = sp_list[0], sp_list[1]

    # Metric 1: Total speaking time per speaker
    dur0 = sum(s.get("end", 0) - s.get("start", 0) for s in segments if s.get("speaker") == sp0)
    dur1 = sum(s.get("end", 0) - s.get("start", 0) for s in segments if s.get("speaker") == sp1)

    # Metric 2: Honorific detection (Korean)
    honorifics = ["습니다", "입니다", "하십시오", "드리", "올리", "여쭤", "모시"]
    hon0 = sum(1 for s in segments if s.get("speaker") == sp0 and any(h in s.get("text", "") for h in honorifics))
    hon1 = sum(1 for s in segments if s.get("speaker") == sp1 and any(h in s.get("text", "") for h in honorifics))

    # Metric 3: First speaker
    first_sp = None
    for seg in segments:
        if seg.get("speaker"):
            first_sp = seg["speaker"]
            break

    # Scoring: each metric votes
    score0_is_caller = 0

    if dur0 > dur1 * 1.3:
        score0_is_caller += 1
    elif dur1 > dur0 * 1.3:
        score0_is_caller -= 1

    if hon0 > hon1:
        score0_is_caller += 2
    elif hon1 > hon0:
        score0_is_caller -= 2

    if first_sp == sp0:
        score0_is_caller += 1
    else:
        score0_is_caller -= 1

    # Determine mapping
    if score0_is_caller > 0:
        mapping = {sp0: caller_name, sp1: MY_NAME}
    elif score0_is_caller < 0:
        mapping = {sp1: caller_name, sp0: MY_NAME}
    else:
        if first_sp:
            mapping = {first_sp: caller_name}
            other = (speakers - {first_sp}).pop()
            mapping[other] = MY_NAME
        else:
            mapping = {}

    for seg in segments:
        sp = seg.get("speaker")
        if sp and sp in mapping:
            seg["speaker"] = mapping[sp]

    return segments


# ── Quality metrics ──────────────────────────────────────────────────────

def log_quality(source_name: str, quality_meta: dict, correction_count: int, elapsed: float):
    """Append quality metrics to JSONL log."""
    QUALITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "source": source_name,
        "elapsed_s": round(elapsed, 1),
        "corrections": correction_count,
        **quality_meta,
    }
    with open(QUALITY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Transcription engine ────────────────────────────────────────────────

def transcribe_only(audio_path):
    """Transcribe with faster-whisper only.

    Returns (segments, duration, quality_meta).
    Uses BatchedInferencePipeline for ~3x throughput on RTX 3090.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(resolve_whisper_model(), device="cuda", compute_type=COMPUTE_TYPE)

    # Try BatchedInferencePipeline for speed
    use_batched = False
    batched_model = None
    try:
        from faster_whisper import BatchedInferencePipeline
        batched_model = BatchedInferencePipeline(model=model)
        use_batched = True
        log("BatchedInferencePipeline enabled (batch_size=16)")
    except Exception as _bat_err:
        log(f"BatchedInferencePipeline unavailable ({_bat_err}), using sequential mode")

    def _run_transcribe(path, offset=0.0):
        if use_batched:
            segs, info = batched_model.transcribe(
                str(path), batch_size=16, beam_size=5,
                language=LANGUAGE, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
        else:
            segs, info = model.transcribe(
                str(path), language=LANGUAGE, beam_size=5,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
            )
        rows = []
        for s in list(segs):
            rows.append({
                "start": float(s.start) + offset,
                "end": float(s.end) + offset,
                "text": s.text,
                "avg_logprob": getattr(s, "avg_logprob", 0.0),
                "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
            })
        return rows, info

    duration_hint = get_audio_duration(audio_path)
    if duration_hint > LONG_AUDIO_CHUNK_THRESHOLD_SEC:
        # Split long audio into chunks to avoid CT2 crashes
        log(f"Long audio chunk mode: {Path(audio_path).name} duration={duration_hint:.1f}s")
        all_rows = []
        infos = []
        with tempfile.TemporaryDirectory(prefix="whisper_chunks_") as tmpdir:
            tmpdir = Path(tmpdir)
            start = 0.0
            chunk_idx = 0
            while start < duration_hint:
                chunk_path = tmpdir / f"chunk_{chunk_idx:04d}.wav"
                cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", str(start), "-t", str(LONG_AUDIO_CHUNK_SEC),
                    "-i", str(audio_path), "-ar", "16000", "-ac", "1", str(chunk_path),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if r.returncode != 0 or not chunk_path.exists() or chunk_path.stat().st_size == 0:
                    raise RuntimeError(f"ffmpeg chunk failed at {start:.1f}s")
                rows, info = _run_transcribe(chunk_path, offset=start)
                all_rows.extend(rows)
                infos.append(info)
                start += LONG_AUDIO_CHUNK_SEC
                chunk_idx += 1
        seg_rows = all_rows
        duration = duration_hint
        language = infos[0].language if infos else LANGUAGE
        language_probability = sum(getattr(i, "language_probability", 0.0) for i in infos) / max(len(infos), 1)
    else:
        seg_rows, info = _run_transcribe(audio_path, offset=0.0)
        duration = info.duration
        language = info.language
        language_probability = info.language_probability

    avg_logprob = sum(s.get("avg_logprob", 0.0) for s in seg_rows) / len(seg_rows) if seg_rows else 0
    quality_meta = {
        "duration": round(duration, 1),
        "language": language,
        "language_prob": round(language_probability, 3),
        "segments": len(seg_rows),
        "avg_logprob": round(avg_logprob, 3),
    }

    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in seg_rows], duration, quality_meta


def align_and_diarize_subprocess(audio_path, segments_json, no_diarize=False):
    """Run align + diarize in a subprocess to avoid DLL crash.

    Calls align_worker.py which imports whisperx in a clean process.
    Returns (segments_list, meta_dict) or (None, meta) on failure.
    """
    worker = Path(__file__).parent / "align_worker.py"
    if not worker.exists():
        log(f"Worker script not found: {worker}")
        return None, {"align_ok": False, "diarize_ok": False, "align_error": "worker script not found", "diarize_error": None, "device": "unknown"}

    tmp_json = OUTPUT_DIR / f"_segments_{Path(audio_path).stem}.json"
    try:
        tmp_json.write_text(json.dumps(segments_json, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log(f"Failed to write segments JSON: {e}")
        return segments_json, {"align_ok": False, "diarize_ok": False, "align_error": str(e), "diarize_error": None, "device": "unknown"}

    cmd = [sys.executable, str(worker),
           "--audio", str(audio_path),
           "--segments", str(tmp_json),
           "--token", str(HF_TOKEN_FILE)]
    if no_diarize:
        cmd.append("--no-diarize")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        out_json = Path(str(tmp_json).replace(".json", "_result.json"))
        if out_json.exists():
            data = json.loads(out_json.read_text(encoding="utf-8"))
            out_json.unlink(missing_ok=True)
            tmp_json.unlink(missing_ok=True)
            if isinstance(data, dict):
                meta = data.get("_meta", {})
                segments = data.get("segments", data)
                return segments, meta
            return data, {}
        else:
            log(f"Worker failed: rc={result.returncode}")
            tmp_json.unlink(missing_ok=True)
            return None, {"align_ok": False, "diarize_ok": False, "align_error": f"rc={result.returncode}", "diarize_error": None, "device": "unknown"}
    except subprocess.TimeoutExpired:
        log(f"Worker timeout for {Path(audio_path).name}")
        tmp_json.unlink(missing_ok=True)
        return None, {"align_ok": False, "diarize_ok": False, "align_error": "timeout", "diarize_error": None, "device": "unknown"}
    except Exception as e:
        log(f"Worker exception: {e}")
        tmp_json.unlink(missing_ok=True)
        return None, {"align_ok": False, "diarize_ok": False, "align_error": str(e), "diarize_error": None, "device": "unknown"}


def main():
    args = parse_args()
    ensure_rules_file()
    log("WhisperX transcription run started")

    if args.file:
        f = Path(args.file)
        if not args.force and (OUTPUT_DIR / f"{f.stem}.txt").exists():
            log(f"Already transcribed (skipping): {f.name}")
            print(f"SKIP: {f.name} already transcribed", flush=True)
            return 0
        pending = [f]
    else:
        pending = get_pending(recent_first=args.recent_first)
        if args.limit:
            pending = pending[:args.limit]

    if not pending:
        log("All files transcribed")
        print("ALL DONE")
        return 0

    gpu_free = get_gpu_free_mb()
    ram_pct = psutil.virtual_memory().percent
    print(f"Pending: {len(pending)} | GPU free: {gpu_free}MB | RAM: {ram_pct}%", flush=True)

    if not args.force and gpu_free < GPU_MIN_FREE_MB:
        print(f"⚠️ GPU low ({gpu_free}MB)", flush=True)
        killed = kill_gpu_hogs()
        if killed:
            time.sleep(2)
            gpu_free = get_gpu_free_mb()
        if gpu_free < GPU_MIN_FREE_MB:
            print("❌ GPU insufficient. Skipping.", flush=True)
            return EXIT_FAILURE

    success = 0
    for i, audio_path in enumerate(pending, 1):
        out_path = OUTPUT_DIR / f"{audio_path.stem}.txt"

        if not args.force:
            gpu_free = get_gpu_free_mb()
            if gpu_free < GPU_MIN_FREE_PER_FILE:
                print(f"[{i}/{len(pending)}] GPU low, stopping", flush=True)
                break

        duration = get_audio_duration(audio_path)
        if duration > MAX_AUDIO_DURATION_SEC:
            print(f"[{i}/{len(pending)}] {audio_path.name} — too long, skipping", flush=True)
            continue

        print(f"[{i}/{len(pending)}] {audio_path.name} ({duration:.0f}s)", flush=True)
        t0 = time.time()

        try:
            caller_name, caller_phone = parse_caller_info(audio_path.stem)

            # Step 1: Transcribe
            segments, audio_dur, quality_meta = transcribe_only(audio_path)
            if not segments:
                raise ValueError("Empty transcription")

            # Steps 2+3: Align + Diarize
            had_diary_fail = False
            if audio_dur >= LONG_AUDIO_FAST_PATH_SEC:
                final_segments = segments
                align_meta = {"align_ok": True, "diarize_ok": True, "device": "skipped_long_audio"}
                print("  ⚡ Long audio fast path: skipping align/diarize", flush=True)
            else:
                final_segments, align_meta = align_and_diarize_subprocess(audio_path, segments, args.no_diarize)
                if final_segments is None:
                    final_segments = segments
                    had_diary_fail = True

            # Speaker name mapping
            if caller_name:
                final_segments = map_speakers(final_segments, caller_name, caller_phone)

            # Format text
            lines = []
            for seg in final_segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                speaker = seg.get("speaker")
                if speaker:
                    lines.append(f"[{seg.get('start',0):.1f}-{seg.get('end',0):.1f}] {speaker}: {text}")
                else:
                    lines.append(f"[{seg.get('start',0):.1f}-{seg.get('end',0):.1f}] {text}")

            text = "\n".join(lines)
            if len(text.strip()) < 10:
                raise ValueError("Near-empty result")

            corrected_text, changes = apply_corrections(text, source=audio_path.name)

            if out_path.exists():
                out_path = OUTPUT_DIR / f"{audio_path.stem}_{datetime.now().strftime('%H%M%S')}.txt"
            safe_write_text(out_path, corrected_text + "\n")
            success += 1

            elapsed = time.time() - t0
            rtf = audio_dur / elapsed if elapsed > 0 else 0

            log_quality(audio_path.name, quality_meta, len(changes) if changes else 0, elapsed)

            diary_flag = " [diarize fail]" if had_diary_fail else ""
            print(f"  ✅ {elapsed:.1f}s ({rtf:.0f}x) corrections={len(changes) if changes else 0}{diary_flag}", flush=True)

        except Exception as e:
            err_str = str(e)
            fail_count = blacklist_add(audio_path.stem, err_str)
            log(f"FAIL ({fail_count}/{MAX_CONSECUTIVE_FAILURES}) {audio_path.name}: {err_str[:80]}")

    gpu_after = get_gpu_free_mb()
    print(f"\nDone: {success}/{len(pending)} | GPU: {gpu_after}MB", flush=True)
    summary = {"success": success, "total": len(pending), "gpu_free_mb": gpu_after}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return EXIT_OK if success > 0 else EXIT_PARTIAL


if __name__ == "__main__":
    sys.exit(main())


