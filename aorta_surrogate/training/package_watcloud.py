"""Create deterministic development-data and source bundles for WATcloud."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

from aorta_surrogate.training.verify_staged_runtime import verify_staged_runtime


SOURCE_PATHS = (
    "aorta_surrogate",
    "configs/watcloud_preop_v1_frozen.json",
    "infra/watcloud",
    "pyproject.toml",
    "tests",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _add_file(archive: tarfile.TarFile, source: Path, archive_name: str) -> None:
    info = archive.gettarinfo(str(source), arcname=archive_name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if source.suffix in {".sh", ".sbatch"} else 0o644
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def _deterministic_tar_gz(
    output_path: Path, rows: list[tuple[Path, str]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source, archive_name in sorted(rows, key=lambda row: row[1]):
                    _add_file(archive, source, archive_name)


def _write_sha_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{_sha256(path)}  {path.name}\n", encoding="ascii")
    return sidecar


def package_watcloud(
    repository_root: Path,
    canonical_root: Path,
    freeze_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    verification = verify_staged_runtime(
        canonical_root,
        freeze_manifest_path,
        allow_locked_source_directories=True,
    )
    freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    data_rows = [
        (
            canonical_root / Path(row["relative_path"]),
            str(PurePosixPath("canonical") / row["relative_path"]),
        )
        for row in freeze["runtime_files"]
    ]
    freeze_archive_name = PurePosixPath(
        "canonical/experiments/watcloud_preop_v1/freeze_manifest.json"
    )
    data_rows.append((freeze_manifest_path, str(freeze_archive_name)))

    source_rows: list[tuple[Path, str]] = []
    for relative in SOURCE_PATHS:
        source = repository_root / Path(relative)
        if source.is_file():
            source_rows.append((source, PurePosixPath(relative).as_posix()))
        elif source.is_dir():
            source_rows.extend(
                (path, path.relative_to(repository_root).as_posix())
                for path in source.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and ".pytest_cache" not in path.parts
            )
        else:
            raise FileNotFoundError(f"source bundle path is missing: {source}")

    data_bundle = output_dir / "watcloud-preop-aaa-v1-data.tar.gz"
    source_bundle = output_dir / "watcloud-preop-aaa-v1-source.tar.gz"
    _deterministic_tar_gz(data_bundle, data_rows)
    _deterministic_tar_gz(source_bundle, source_rows)
    data_sidecar = _write_sha_sidecar(data_bundle)
    source_sidecar = _write_sha_sidecar(source_bundle)
    report = {
        "schema_version": "1.0.0",
        "status": "watcloud_bundles_created",
        "contract_sha256": freeze["contract_sha256"],
        "runtime_tree_sha256": freeze["runtime_tree_sha256"],
        "staged_runtime_verification": verification,
        "bundles": {
            "data": {
                "path": str(data_bundle),
                "sha256": _sha256(data_bundle),
                "bytes": data_bundle.stat().st_size,
                "sidecar": str(data_sidecar),
            },
            "source": {
                "path": str(source_bundle),
                "sha256": _sha256(source_bundle),
                "bytes": source_bundle.stat().st_size,
                "sidecar": str(source_sidecar),
            },
        },
    }
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package_watcloud(
                args.repository_root,
                args.canonical_root,
                args.freeze_manifest,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
