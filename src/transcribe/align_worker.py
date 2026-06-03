"""WhisperX align + diarize worker.

Runs in a separate process to avoid CTranslate2/whisperx DLL crash.

Input: --audio <path> --segments <json_file> [--token <hf_token_file>] [--no-diarize]
Output: writes <segments_file>_result.json with aligned/diarized segments.

This script ONLY imports whisperx (no faster_whisper/ctranslate2).
"""

import argparse
import gc
import json
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

# ── Logging ──────────────────────────────────────────────────────────────
LOG = Path(os.environ.get("ALIGN_LOG", "logs/align_whisperx.log"))


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        log.warning(f"{ts} log write failed: {exc}")
    print(line, flush=True)


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
    except Exception as exc:
        log(f"GPU memory query failed: {exc}")
    return -1


def resolve_device(requested: str = "cuda") -> str:
    """Return 'cuda' if available and has enough free memory, else 'cpu'."""
    if requested == "cuda" and torch.cuda.is_available():
        free = get_gpu_free_mb()
        if free > 0 and free < 1024:
            log(f"GPU free={free}MB < 1024MB threshold, falling back to CPU")
            return "cpu"
        return "cuda"
    if requested == "cuda" and not torch.cuda.is_available():
        log("CUDA not available, falling back to CPU")
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
            log(f"load_audio timeout (attempt {attempt}/{retries})")
        except subprocess.CalledProcessError as e:
            last_err = e
            log(f"load_audio ffmpeg error rc={e.returncode} (attempt {attempt}/{retries})")
        except Exception as e:
            last_err = e
            log(f"load_audio unexpected error: {e} (attempt {attempt}/{retries})")
        if attempt < retries:
            time.sleep(1)
    raise RuntimeError(f"load_audio failed after {retries} attempts: {last_err}")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="WhisperX align + diarize worker")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--segments", required=True, help="JSON file with segments")
    parser.add_argument("--token", help="Path to HF token file")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    args = parser.parse_args()

    device = resolve_device("cuda")
    log(f"Worker start: audio={Path(args.audio).name} device={device}")

    # Load segments
    try:
        segments = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    except Exception as e:
        log(f"FATAL: cannot read segments file: {e}")
        return 1

    out_path = Path(str(args.segments).replace(".json", "_result.json"))

    align_ok = False
    diarize_ok = False
    align_error: str | None = None
    diarize_error: str | None = None

    # ---- Step 1: Align ----
    segs_aligned = segments
    try:
        import whisperx
        log("Loading audio for alignment...")
        audio = load_audio(args.audio)
        log("Loading align model...")
        align_model, metadata = whisperx.load_align_model(language_code="ko", device=device)
        log(f"Aligning {len(segments)} segments...")
        t0 = time.time()
        result = whisperx.align(segments, align_model, metadata, audio, device)
        elapsed = time.time() - t0
        segs_aligned = result.get("segments", []) if isinstance(result, dict) else result.segments
        align_ok = True
        log(f"Align OK: {len(segs_aligned)} segments in {elapsed:.1f}s")
        del align_model, audio
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError as e:
        align_error = f"whisperx import failed: {e}"
        log(f"ALIGN_FAIL: {align_error}")
    except RuntimeError as e:
        align_error = str(e)
        log(f"ALIGN_FAIL: {align_error}")
        # If CUDA OOM, try once on CPU
        if device == "cuda" and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
            log("Retrying alignment on CPU...")
            try:
                import whisperx
                gc.collect()
                torch.cuda.empty_cache()
                audio = load_audio(args.audio)
                align_model, metadata = whisperx.load_align_model(language_code="ko", device="cpu")
                t0 = time.time()
                result = whisperx.align(segments, align_model, metadata, audio, "cpu")
                elapsed = time.time() - t0
                segs_aligned = result.get("segments", []) if isinstance(result, dict) else result.segments
                align_ok = True
                align_error = None
                log(f"Align OK (CPU retry): {len(segs_aligned)} segments in {elapsed:.1f}s")
                del align_model, audio
                gc.collect()
            except Exception as e2:
                align_error = f"CPU retry also failed: {e2}"
                log(f"ALIGN_FAIL (CPU retry): {align_error}")
    except Exception as e:
        align_error = str(e)
        log(f"ALIGN_FAIL: {align_error}")

    # ---- Step 2: Diarize ----
    segs_final = segs_aligned
    if not args.no_diarize and args.token:
        try:
            from whisperx.diarize import DiarizationPipeline
            hf_token = Path(args.token).read_text(encoding="utf-8").strip()
            log("Loading diarization model...")
            dm = DiarizationPipeline(token=hf_token, device=device)
            log("Running diarization (min=2, max=2 speakers)...")
            t0 = time.time()
            diar = dm(args.audio, min_speakers=2, max_speakers=2)
            elapsed = time.time() - t0
            result_d = whisperx.assign_word_speakers(diar, {"segments": segs_aligned})
            segs_final = result_d.get("segments", segs_aligned)
            diarize_ok = True
            log(f"Diarize OK in {elapsed:.1f}s")
            del dm, diar
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            diarize_error = str(e)
            log(f"DIARIZE_FAIL: {diarize_error}")
    elif args.no_diarize:
        log("Diarization skipped (--no-diarize)")
    elif not args.token:
        log("Diarization skipped (no --token)")

    # ---- Serialize output ----
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

    # Atomic write
    try:
        tmp_path = out_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(result_payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(out_path)
    except Exception as e:
        log(f"FATAL: cannot write result: {e}")
        return 1

    status_parts = []
    status_parts.append("align=ok" if align_ok else "align=fail")
    if not args.no_diarize and args.token:
        status_parts.append("diarize=ok" if diarize_ok else "diarize=fail")
    log(f"OK: {len(output)} segments ({', '.join(status_parts)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
