import json
from datetime import datetime, timedelta, timezone


def test_health_ignores_timestamp_suffixed_recheck_transcripts(tmp_path, monkeypatch):
    from src.pipeline import health_check as pipeline_health_check

    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    state_file = tmp_path / "state.json"
    blacklist_file = tmp_path / "blacklist.json"
    todos_file = tmp_path / "todos.json"
    audio_dir.mkdir()
    transcript_dir.mkdir()

    stem = "caller_alpha_20260510101010"
    (audio_dir / f"{stem}.m4a").write_bytes(b"x" * 2048)
    (transcript_dir / f"{stem}.txt").write_text("기존 정상 전사본입니다. 내용 충분함.", encoding="utf-8")
    # Diarization recheck outputs use an extra HHMMSS suffix and should not be
    # treated as a new unprocessed call transcript.
    (transcript_dir / f"{stem}_003108.txt").write_text("[상대방] 재점검 전사본입니다. 내용 충분함.", encoding="utf-8")

    processed = {stem: {"processed_at": datetime.now(timezone(timedelta(hours=9))).isoformat()}}
    state_file.write_text(json.dumps({"processed_transcripts": processed}, ensure_ascii=False), encoding="utf-8")
    blacklist_file.write_text("{}", encoding="utf-8")
    todos_file.write_text(json.dumps({"todos": {}}, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(pipeline_health_check, "AUDIO_DIR", audio_dir)
    monkeypatch.setattr(pipeline_health_check, "TRANSCRIPT_DIR", transcript_dir)
    monkeypatch.setattr(pipeline_health_check, "STATE_FILE", state_file)
    monkeypatch.setattr(pipeline_health_check, "BLACKLIST_FILE", blacklist_file)
    monkeypatch.setattr(pipeline_health_check, "PERSISTENT_TODOS", todos_file)

    assert pipeline_health_check.canonical_transcript_stem("caller_alpha_20260506130327_004143") == "caller_alpha_20260506130327"
    assert pipeline_health_check.main() == 0



