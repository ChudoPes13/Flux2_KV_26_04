#!/usr/bin/env python3
"""Check Hugging Face access, BFL model download size, disk, and GPU capacity.

The check is read-only: it never downloads model files or writes an access token.

Target: black-forest-labs/FLUX.2-klein-9b-kv on RTX 5090 (32 GiB VRAM).
No ApacheOne, no third-party text encoder — NVFP4 checkpoint is produced by this
project itself via ModelOpt in Sprint 004.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GIB = 1024**3
OFFICIAL_BFL_MINIMUM_VRAM_GIB = 29
WORKING_HEADROOM_GIB = 15
DOWNLOAD_OVERHEAD = 1.15
MODEL_REPOSITORIES = {
    "bfl_primary": "black-forest-labs/FLUX.2-klein-9b-kv",
}


def bytes_to_gib(value: int) -> float:
    return round(value / GIB, 2)


def sibling_size(sibling: Any) -> int:
    """Return a file's Hub-reported size across current hub client versions."""
    direct_size = getattr(sibling, "size", None)
    if isinstance(direct_size, int):
        return direct_size
    lfs = getattr(sibling, "lfs", None)
    lfs_size = getattr(lfs, "size", None)
    return lfs_size if isinstance(lfs_size, int) else 0


def get_repository_sizes() -> tuple[bool, list[dict[str, object]], str | None]:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=True)
        api.whoami()
        repositories: list[dict[str, object]] = []
        for role, repo_id in MODEL_REPOSITORIES.items():
            info = api.model_info(repo_id, files_metadata=True)
            sizes = [sibling_size(sibling) for sibling in info.siblings or []]
            repositories.append(
                {
                    "role": role,
                    "repo_id": repo_id,
                    "file_count": len(sizes),
                    "bytes": sum(sizes),
                    "gib": bytes_to_gib(sum(sizes)),
                }
            )
        return True, repositories, None
    except Exception as error:  # noqa: BLE001
        return False, [], f"{type(error).__name__}: {error}"


def gpu_snapshot() -> dict[str, object]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "free_gib": bytes_to_gib(free_bytes),
            "total_gib": bytes_to_gib(total_bytes),
            "official_bfl_minimum_gib": OFFICIAL_BFL_MINIMUM_VRAM_GIB,
            "meets_official_bfl_requirement": total_bytes >= OFFICIAL_BFL_MINIMUM_VRAM_GIB * GIB,
        }
    except Exception as error:  # noqa: BLE001
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/diagnostics/model_readiness.json")
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    authenticated, repositories, error = get_repository_sizes()
    model_root = args.model_root.resolve()
    disk = shutil.disk_usage(model_root)
    total_download_bytes = sum(repository["bytes"] for repository in repositories)
    required_bytes = int(total_download_bytes * DOWNLOAD_OVERHEAD) + WORKING_HEADROOM_GIB * GIB
    storage_ready = authenticated and disk.free >= required_bytes
    gpu = gpu_snapshot()
    warnings: list[str] = []
    if gpu.get("available") and not gpu.get("meets_official_bfl_requirement"):
        warnings.append(
            "The official BFL model documents approximately 29 GiB VRAM; this GPU does not meet "
            "that published requirement. BF16 baseline may OOM; consider switching to a 32 GiB "
            "Blackwell GPU (RTX 5090) before proceeding."
        )
    if gpu.get("available") and gpu.get("capability") and gpu["capability"][0] < 10:
        warnings.append(
            "This GPU is not Blackwell-class (SM < 10.0). NVFP4 acceptance requires Blackwell."
        )

    checks = {
        "huggingface_authenticated_and_accessible": authenticated,
        "model_metadata_available": authenticated
        and len(repositories) == len(MODEL_REPOSITORIES),
        "storage_ready": storage_ready,
        "gpu_available": bool(gpu.get("available")),
    }
    report = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "model_root": str(model_root),
        "repositories": repositories,
        "download_plan": {
            "repository_total_gib": bytes_to_gib(total_download_bytes),
            "download_overhead_factor": DOWNLOAD_OVERHEAD,
            "working_headroom_gib": WORKING_HEADROOM_GIB,
            "required_free_gib": bytes_to_gib(required_bytes),
            "available_free_gib": bytes_to_gib(disk.free),
        },
        "gpu": gpu,
        "checks": checks,
        "warnings": warnings,
        "error": error,
        "status": "pass" if all(checks.values()) else "blocked",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not args.strict or report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
