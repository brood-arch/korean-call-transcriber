def test_sync_obsidian_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    from src.sync import sync_obsidian

    source = tmp_path / "transcripts"
    vault = tmp_path / "vault"
    state = tmp_path / "state.json"
    source.mkdir()
    (source / "Acme_01012345678_20260601120000.txt").write_text("가" * 120, encoding="utf-8")

    monkeypatch.setattr(sync_obsidian, "SOURCE_DIR", source)
    monkeypatch.setattr(sync_obsidian, "OBSIDIAN_VAULT", vault)
    monkeypatch.setattr(sync_obsidian, "TRANSCRIPTS_DIR", vault / "transcripts")
    monkeypatch.setattr(sync_obsidian, "COUNTERPARTY_DIR", vault / "contacts")
    monkeypatch.setattr(sync_obsidian, "COUNTERPARTY_INDEX", vault / "contacts" / "contact_index.md")
    monkeypatch.setattr(sync_obsidian, "STATE_FILE", state)
    monkeypatch.setattr("sys.argv", ["sync_obsidian", "--dry-run"])

    sync_obsidian.main()

    assert "처리 완료: 1개" in capsys.readouterr().out
    assert not (vault / "transcripts" / "2026-06-01_Acme.md").exists()
    assert not state.exists()


def test_update_counterparty_file_creates_and_appends(tmp_path, monkeypatch):
    from src.sync import sync_obsidian

    contacts = tmp_path / "contacts"
    monkeypatch.setattr(sync_obsidian, "COUNTERPARTY_DIR", contacts)

    sync_obsidian.update_counterparty_file("Acme", "2026-06-01_Acme.md", "2026-06-01")
    sync_obsidian.update_counterparty_file("Acme", "2026-06-02_Acme.md", "2026-06-02")

    content = (contacts / "Acme.md").read_text(encoding="utf-8")
    assert "[2026-06-01]" in content
    assert "[2026-06-02]" in content


def test_update_counterparty_index_adds_new_entries(tmp_path, monkeypatch):
    from src.sync import sync_obsidian

    index = tmp_path / "contact_index.md"
    index.write_text("# Contacts\n", encoding="utf-8")
    monkeypatch.setattr(sync_obsidian, "COUNTERPARTY_INDEX", index)

    sync_obsidian.update_counterparty_index({"Beta"})

    content = index.read_text(encoding="utf-8")
    assert "### Beta" in content
    assert "[[Beta]]" in content


def test_load_state_handles_bad_json(tmp_path, monkeypatch):
    from src.sync import sync_obsidian

    state = tmp_path / "state.json"
    state.write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(sync_obsidian, "STATE_FILE", state)

    assert sync_obsidian.load_state() == {"processed": {}, "last_run": None}


def test_parse_filename_rejects_invalid_name():
    from src.sync.sync_obsidian import parse_filename

    assert parse_filename("bad.txt") is None
