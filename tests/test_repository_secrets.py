from pathlib import Path

from scripts.check_repository_secrets import scan_paths


def test_secret_scanner_reports_rule_without_secret_value(tmp_path: Path) -> None:
    safe = tmp_path / "safe.txt"
    unsafe = tmp_path / "unsafe.txt"
    safe.write_text("token is supplied through the instance role\n", encoding="utf-8")
    unsafe.write_text("api_" + "key = \"this-is-not-safe\"\n", encoding="utf-8")
    assert scan_paths(tmp_path, [safe.name]) == []
    assert scan_paths(tmp_path, [unsafe.name]) == [(unsafe.name, "assigned_secret")]
