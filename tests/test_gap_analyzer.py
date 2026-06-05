import json


def test_transcription_gap_analyzer_taxonomy_and_derivatives(tmp_path):
    from src.queue import gap_analyzer as a

    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    state_dir = tmp_path / "state"
    integrated_dir = state_dir / "integrated_extraction"
    log_dir = tmp_path / "logs"
    for p in (audio_dir, transcript_dir, state_dir, integrated_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    ok = "account_call_20260510101010"
    missing = "missing_call_20260510111111"
    blacklisted = "excluded_call_20260510121212"
    failed = "failed_call_20260510131313"

    for stem in (ok, missing, blacklisted, failed):
        (audio_dir / f"{stem}.m4a").write_bytes(b"x" * 2048)
    (transcript_dir / f"{ok}.txt").write_text("정상 전사본입니다. 내용 충분함.", encoding="utf-8")
    (transcript_dir / f"{ok}_003108.txt").write_text("재점검 파생본입니다. 내용 충분함.", encoding="utf-8")

    blacklist_file = state_dir / "transcribe_blacklist.json"
    blacklist_file.write_text(
        json.dumps(
            {
                blacklisted: {"failures": 3, "blacklisted_at": "2026-05-08T00:00:00+09:00"},
                failed: {"failures": 1, "blacklisted_at": None},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "chroma_index_state.json").write_text(
        json.dumps(
            {"files": {str(transcript_dir / f"{ok}.txt"): {}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (state_dir / "sync_transcripts_state.json").write_text(
        json.dumps(
            {"processed": {f"{ok}.txt": {}}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (integrated_dir / "batch_0000.json").write_text(json.dumps({"files": [ok]}, ensure_ascii=False), encoding="utf-8")
    (log_dir / "transcribe_vv.log").write_text(
        f"2026-05-01 Speaker diarization JSON parse failed for {ok}.m4a: bad json\n",
        encoding="utf-8",
    )

    report = a.analyze(
        workspace=tmp_path,
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        blacklist_file=blacklist_file,
        chroma_state_file=state_dir / "chroma_index_state.json",
        obsidian_state_file=state_dir / "sync_transcripts_state.json",
        integrated_extraction_dir=integrated_dir,
        transcribe_log=log_dir / "transcribe_vv.log",
    )

    assert a.canonical_transcript_stem(f"{ok}_003108") == ok
    assert report["category_counts"]["missing_transcript"] == 1
    assert report["category_counts"]["blacklisted"] == 1
    assert report["category_counts"]["transcription_failed"] == 1
    assert report["category_counts"]["derived_excluded"] == 1
    assert report["category_counts"]["diarization_failed"] == 1
    assert report["counts"]["exact_transcript_gap_count"] == 3
    assert report["exit_code"] == 2
    assert all(row["reason"] != "derived_excluded" for row in report["action_queue"])



