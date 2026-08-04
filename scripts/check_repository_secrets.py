"""Fail when tracked text files contain common credential material."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


PATTERNS = {
    "private_key": re.compile(
        rb"-----BEGIN " + rb"(?:RSA|OPENSSH|EC)" + rb" PRIVATE KEY-----"
    ),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "assigned_secret": re.compile(
        rb"(?i)\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"
    ),
}


def scan_paths(root: Path, relative_paths: list[str]) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        content = path.read_bytes()
        if b"\x00" in content:
            continue
        for rule, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append((relative, rule))
    return findings


def tracked_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\x00") if path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    findings = scan_paths(root, tracked_paths(root))
    if findings:
        for relative, rule in findings:
            print(f"POTENTIAL SECRET: {relative} ({rule})")
        raise SystemExit(1)
    print("Tracked-file credential scan passed.")


if __name__ == "__main__":
    main()
