#!/usr/bin/env python3
"""WhisperX transcribe-only worker.

Called by batch_transcribe_whisperx.py in --force mode.
Runs faster-whisper transcription in an isolated subprocess to avoid
CTranslate2 DLL cleanup crashes.

Output: writes {stem}.segments.json and {stem}.meta.json to OUTPUT_DIR.
Exit 0 = success, exit 2 = error.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# --- Config (same as parent script) ---
SOURCE_DIR = Path(os.environ.get("KCT_AUDIO_DIR", "data/audio"))
OUTPUT_DIR = Path(os.environ.get("KCT_TRANSCRIPT_DIR", "output/transcripts"))

LONG_AUDIO_CHUNK_THRESHOLD_SEC = 600
LONG_AUDIO_CHUNK_SEC = 300
COMPUTE_TYPE = "float16"
LANGUAGE = "ko"


# ffmpeg setup — ensure Scripts dir is in PATH (has ffmpeg.exe)
_scripts_dir = str(Path(sys.executable).parent)
os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")


def get_audio_duration(path):
    """Get audio duration via ffmpeg (ffprobe may not be in PATH)."""
    # Try ffprobe first, fall back to ffmpeg
    for cmd_name in ["ffprobe", "ffmpeg"]:
        if cmd_name == "ffprobe":
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "csv=p=0", str(path)]
        else:
            cmd = ["ffmpeg", "-i", str(path), "-f", "null", "-"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if cmd_name == "ffprobe" and r.returncode == 0:
                return float(r.stdout.strip())
            elif cmd_name == "ffmpeg":
                # Parse "  Duration: HH:MM:SS.mm" from stderr
                import re
                m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
                if m:
                    return int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
        except (subprocess.SubprocessError, OSError) as exc:  # noqa: BLE001
            log.warning("duration probe failed for %s: %s", cmd_name, exc)
            continue
    return 0.0


def transcribe_file(audio_path, model_path, compute_type, language, beam_size, output_dir):
    """Transcribe a single audio file with faster-whisper.

    Uses BatchedInferencePipeline for ~3x throughput on RTX 3090.
    Falls back to sequential model.transcribe() if BatchedInferencePipeline is unavailable.
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_path, device="cuda", compute_type=compute_type)

    # Try BatchedInferencePipeline for speed; fall back to sequential on import failure
    use_batched = False
    batched_model = None
    try:
        from faster_whisper import BatchedInferencePipeline
        batched_model = BatchedInferencePipeline(model=model)
        use_batched = True
        log.info("BatchedInferencePipeline enabled (batch_size=16)", flush=True)
    except (ImportError, RuntimeError, AttributeError) as _bat_err:  # noqa: BLE001
        log.info(f"BatchedInferencePipeline unavailable ({_bat_err}), using sequential mode", flush=True)

    def _run_batched(path, offset=0.0):
        segs, info = batched_model.transcribe(
            str(path), batch_size=16, beam_size=beam_size,
            language=language, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        rows = []
        for s in list(segs):
            rows.append({
                "start": float(s.start) + offset,
                "end": float(s.end) + offset,
                "text": s.text,
            })
        return rows, info

    def _run_sequential(path, offset=0.0):
        segs, info = model.transcribe(
            str(path), language=language, beam_size=beam_size,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
        )
        rows = []
        for s in list(segs):
            rows.append({
                "start": float(s.start) + offset,
                "end": float(s.end) + offset,
                "text": s.text,
            })
        return rows, info

    _run = _run_batched if use_batched else _run_sequential

    duration_hint = get_audio_duration(audio_path)
    stem = Path(audio_path).stem

    if duration_hint > LONG_AUDIO_CHUNK_THRESHOLD_SEC:
        all_rows = []
        infos = []
        with tempfile.TemporaryDirectory(prefix="wx_chunk_") as tmpdir:
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
                    break
                rows, info = _run(chunk_path, offset=start)
                all_rows.extend(rows)
                infos.append(info)
                start += LONG_AUDIO_CHUNK_SEC
                chunk_idx += 1
        seg_rows = all_rows
        duration = duration_hint
    else:
        seg_rows, info = _run(audio_path, offset=0.0)
        duration = info.duration

    quality_meta = {
        "duration": round(duration, 1),
        "language": language,
        "segments": len(seg_rows),
    }

    # Write outputs (atomic: write to tmp then rename)
    segments_file = Path(output_dir) / f"{stem}.segments.json"
    meta_file = Path(output_dir) / f"{stem}.meta.json"

    for target, payload in [(segments_file, seg_rows), (meta_file, quality_meta)]:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)

    log.info(f"OK: {stem} — {len(seg_rows)} segments, {duration:.0f}s", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()

    try:
        return transcribe_file(
            args.audio, args.model, args.compute_type,
            args.language, args.beam_size, args.output_dir,
        )
    except Exception as e:  # noqa: BLE001 — main() top-level catch
        log.error("transcribe_file failed: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())


