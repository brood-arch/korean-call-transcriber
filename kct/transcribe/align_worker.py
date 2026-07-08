"""WhisperX align + diarize worker.

Runs in a separate process to avoid CTranslate2/whisperx DLL crash.

Input: --audio <path> --segments <json_file> [--token <hf_token_file>] [--no-diarize]
Output: writes <segments_file>_result.json with aligned/diarized segments.

This script ONLY imports whisperx (no faster_whisper/ctranslate2).
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)

# ── Logging ──────────────────────────────────────────────────────────────
LOG = Path(os.environ.get("ALIGN_LOG", "logs/align_whisperx.log"))


def _log_to_file(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:  # noqa: BLE001
        log.warning(f"{ts} log write failed: {exc}")
    log.info(line)


# ── GPU helpers ──────────────────────────────────────────────────────────

def get_gpu_free_mb() -> int:
    """Query free GPU memory in MB. Returns -1 if query fails."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0])
    except (subprocess.SubprocessError, OSError) as exc:  # noqa: BLE001
        _log_to_file(f"GPU memory query failed: {exc}")
    return -1


def resolve_device(requested: str = "cuda") -> str:
    """Return 'cuda' if available and has enough free memory, else 'cpu'."""
    if requested == "cuda" and torch.cuda.is_available():
        free = get_gpu_free_mb()
        if free > 0 and free < 1024:
            _log_to_file(f"GPU free={free}MB < 1024MB threshold, falling back to CPU")
            return "cpu"
        return "cuda"
    if requested == "cuda" and not torch.cuda.is_available():
        _log_to_file("CUDA not available, falling back to CPU")
        return "cpu"
    return requested


# ── Audio loading ────────────────────────────────────────────────────────

def load_audio(file: str, sr: int = 16000, retries: int = 2) -> np.ndarray:
    """Load audio via ffmpeg with retry on transient failure."""
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file,
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        except subprocess.TimeoutExpired as e:
            last_err = e
            _log_to_file(f"load_audio timeout (attempt {attempt}/{retries})")
        except subprocess.CalledProcessError as e:
            last_err = e
            _log_to_file(f"load_audio ffmpeg error rc={e.returncode} (attempt {attempt}/{retries})")
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log_to_file(f"load_audio unexpected error: {e} (attempt {attempt}/{retries})")
        if attempt < retries:
            time.sleep(1)
    raise RuntimeError(f"load_audio failed after {retries} attempts: {last_err}")


# ── Main ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="WhisperX align + diarize worker")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--segments", required=True, help="JSON file with segments")
    parser.add_argument("--token", help="Path to HF token file")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    return parser.parse_args()


def _load_segments(segments_path: str) -> tuple[list, str | None]:
    """Load segments JSON. Returns (segments, error)."""
    try:
        return json.loads(Path(segments_path).read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, OSError) as e:  # noqa: BLE001
        _log_to_file(f"FATAL: cannot read segments file: {e}")
        return [], str(e)


def _run_alignment(
    segments: list, audio_path: str, device: str,
) -> tuple[list, bool, str | None]:
    """Run WhisperX alignment with CPU fallback on CUDA OOM.

    Returns (aligned_segments, success, error_message).
    """
    try:
        import whisperx
        _log_to_file("Loading audio for alignment...")
        audio = load_audio(audio_path)
        _log_to_file("Loading align model...")
        align_model, metadata = whisperx.load_align_model(language_code="ko", device=device)
        _log_to_file(f"Aligning {len(segments)} segments...")
        t0 = time.time()
        result = whisperx.align(segments, align_model, metadata, audio, device)
        elapsed = time.time() - t0
        segs = result.get("segments", []) if isinstance(result, dict) else result.segments
        _log_to_file(f"Align OK: {len(segs)} segments in {elapsed:.1f}s")
        del align_model, audio
        gc.collect()
        torch.cuda.empty_cache()
        return segs, True, None
    except ImportError as e:
        msg = f"whisperx import failed: {e}"
        _log_to_file(f"ALIGN_FAIL: {msg}")
        return segments, False, msg
    except RuntimeError as e:
        msg = str(e)
        _log_to_file(f"ALIGN_FAIL: {msg}")
        if device == "cuda" and ("out of memory" in msg.lower() or "cuda" in msg.lower()):
            return _retry_alignment_cpu(segments, audio_path, msg)
        return segments, False, msg
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        _log_to_file(f"ALIGN_FAIL: {msg}")
        return segments, False, msg


def _retry_alignment_cpu(
    segments: list, audio_path: str, original_error: str,
) -> tuple[list, bool, str | None]:
    """Retry alignment on CPU after CUDA failure."""
    _log_to_file("Retrying alignment on CPU...")
    try:
        import whisperx
        gc.collect()
        torch.cuda.empty_cache()
        audio = load_audio(audio_path)
        align_model, metadata = whisperx.load_align_model(language_code="ko", device="cpu")
        t0 = time.time()
        result = whisperx.align(segments, align_model, metadata, audio, "cpu")
        elapsed = time.time() - t0
        segs = result.get("segments", []) if isinstance(result, dict) else result.segments
        _log_to_file(f"Align OK (CPU retry): {len(segs)} segments in {elapsed:.1f}s")
        del align_model, audio
        gc.collect()
        return segs, True, None
    except Exception as e2:  # noqa: BLE001
        msg = f"CPU retry also failed: {e2}"
        _log_to_file(f"ALIGN_FAIL (CPU retry): {msg}")
        return segments, False, msg


def _run_diarization(
    segs_aligned: list, audio_path: str, token_path: str, device: str,
) -> tuple[list, bool, str | None]:
    """Run speaker diarization. Returns (segments, success, error)."""
    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline
        hf_token = Path(token_path).read_text(encoding="utf-8").strip()
        _log_to_file("Loading diarization model...")
        dm = DiarizationPipeline(token=hf_token, device=device)
        _log_to_file("Running diarization (min=2, max=2 speakers)...")
        t0 = time.time()
        diar = dm(audio_path, min_speakers=2, max_speakers=2)
        elapsed = time.time() - t0
        result_d = whisperx.assign_word_speakers(diar, {"segments": segs_aligned})
        segs_final = result_d.get("segments", segs_aligned)
        _log_to_file(f"Diarize OK in {elapsed:.1f}s")
        del dm, diar
        gc.collect()
        torch.cuda.empty_cache()
        return segs_final, True, None
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        _log_to_file(f"DIARIZE_FAIL: {msg}")
        return segs_aligned, False, msg


def _serialize_output(segs_final: list) -> list:
    """Convert segments to output format."""
    output = []
    for seg in segs_final:
        item = {
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", "").strip(),
        }
        if "speaker" in seg:
            item["speaker"] = seg["speaker"]
        output.append(item)
    return output


def _write_result(out_path: Path, payload: dict) -> bool:
    """Atomic write result JSON. Returns True on success."""
    try:
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(out_path)
        return True
    except OSError as e:  # noqa: BLE001
        _log_to_file(f"FATAL: cannot write result: {e}")
        return False


def main() -> int:
    args = _parse_args()
    device = resolve_device("cuda")
    _log_to_file(f"Worker start: audio={Path(args.audio).name} device={device}")

    segments, load_err = _load_segments(args.segments)
    if load_err:
        return 1

    out_path = Path(str(args.segments).replace(".json", "_result.json"))

    # Step 1: Align
    segs_aligned, align_ok, align_error = _run_alignment(
        segments, args.audio, device,
    )

    # Step 2: Diarize
    segs_final = segs_aligned
    diarize_ok = False
    diarize_error: str | None = None
    if args.no_diarize:
        _log_to_file("Diarization skipped (--no-diarize)")
    elif not args.token:
        _log_to_file("Diarization skipped (no --token)")
    else:
        segs_final, diarize_ok, diarize_error = _run_diarization(
            segs_aligned, args.audio, args.token, device,
        )

    # Serialize & write
    output = _serialize_output(segs_final)
    result_payload = {
        "segments": output,
        "_meta": {
            "align_ok": align_ok,
            "diarize_ok": diarize_ok,
            "align_error": align_error,
            "diarize_error": diarize_error,
            "device": device,
        },
    }
    if not _write_result(out_path, result_payload):
        return 1

    status_parts = []
    status_parts.append("align=ok" if align_ok else "align=fail")
    if not args.no_diarize and args.token:
        status_parts.append("diarize=ok" if diarize_ok else "diarize=fail")
    _log_to_file(f"OK: {len(output)} segments ({', '.join(status_parts)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
