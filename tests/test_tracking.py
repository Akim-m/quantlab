import json

from quantlab.tracking import log_run


def test_log_run_appends_record_with_metadata(tmp_path) -> None:
    path = tmp_path / "log.jsonl"

    log_run({"strategy": "equal_weight", "status": "success"}, path)
    log_run({"strategy": "inverse_vol_63d", "status": "failed"}, path)

    lines = path.read_text().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["strategy"] == "equal_weight"
    assert "timestamp" in first
    assert "git_commit" in first
    assert "git_dirty" in first
