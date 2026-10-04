from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs" / "artifacts_v1.json"


class ArtifactVerificationError(RuntimeError):
    """Raised when a configured research artifact does not match its manifest identity."""


def load_artifact_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ArtifactVerificationError(f"Unsupported artifact manifest schema: {manifest.get('schema_version')!r}")
    return manifest


def artifact_entry(artifact_id: str) -> Dict[str, Any]:
    artifacts = load_artifact_manifest().get("artifacts", {})
    entry = artifacts.get(artifact_id)
    if not isinstance(entry, dict):
        raise ArtifactVerificationError(f"Unknown artifact identity: {artifact_id}")
    return dict(entry)


def artifact_path(artifact_id: str) -> Path:
    local_path = artifact_entry(artifact_id).get("local_path")
    if not local_path:
        raise ArtifactVerificationError(f"Artifact {artifact_id} has no local placement.")
    return PROJECT_ROOT / str(local_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def manifest_artifact_for_path(path: Path) -> Optional[str]:
    resolved = path.resolve()
    for artifact_id, entry in load_artifact_manifest().get("artifacts", {}).items():
        local_path = entry.get("local_path")
        if local_path and (PROJECT_ROOT / str(local_path)).resolve() == resolved:
            return str(artifact_id)
    return None


def verify_local_artifact(artifact_id: str, path: Optional[Path] = None) -> Dict[str, Any]:
    entry = artifact_entry(artifact_id)
    target = path or artifact_path(artifact_id)
    expected = str(entry.get("sha256") or "").upper()
    if not target.is_file():
        raise ArtifactVerificationError(f"{artifact_id} is unavailable: expected file at {target}")
    if not expected:
        raise ArtifactVerificationError(f"{artifact_id} has no expected SHA256 in the v1 manifest.")
    actual = sha256_file(target)
    if actual != expected:
        raise ArtifactVerificationError(
            f"{artifact_id} SHA256 mismatch at {target}: expected {expected}, got {actual}"
        )
    return {"artifact_id": artifact_id, "path": str(target), "sha256": actual, "verified": True}


def verify_qwen_snapshot(path: Path) -> Dict[str, Any]:
    """Verify canonical Qwen provenance from an immutable HF cache snapshot path."""
    entry = artifact_entry("qwen_merged")
    repository = str(entry["repository"])
    revision = str(entry["revision"])
    resolved = path.resolve()
    expected_repo_dir = "models--" + repository.replace("/", "--")
    valid_layout = (
        resolved.is_dir()
        and resolved.name == revision
        and resolved.parent.name == "snapshots"
        and resolved.parent.parent.name == expected_repo_dir
    )
    if not valid_layout:
        raise ArtifactVerificationError(
            "qwen_merged provenance is unverified: expected an HF cache snapshot "
            f"for {repository} at immutable revision {revision}, got {resolved}"
        )
    return {
        "artifact_id": "qwen_merged",
        "path": str(resolved),
        "repository": repository,
        "revision": revision,
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a CySent v1 local research artifact.")
    parser.add_argument("artifact_id", choices=sorted(load_artifact_manifest().get("artifacts", {})))
    parser.add_argument("--path", type=Path, help="Override local path; required for qwen_merged.")
    args = parser.parse_args()
    if args.artifact_id == "qwen_merged":
        if args.path is None:
            parser.error("qwen_merged requires --path to an immutable HF cache snapshot")
        result = verify_qwen_snapshot(args.path)
    else:
        result = verify_local_artifact(args.artifact_id, args.path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
