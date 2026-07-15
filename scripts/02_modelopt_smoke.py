#!/usr/bin/env python3
"""Validate the isolated NVIDIA ModelOpt environment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("+")[0].split(".")[:3])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/modelopt_smoke.json"),
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python": ".".join(map(str, sys.version_info[:3])),
        "checks": {},
    }
    checks: dict[str, bool] = report["checks"]  # type: ignore[assignment]

    try:
        import modelopt
        import modelopt.torch  # noqa: F401
        import torch
        import transformers

        report["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        report["transformers"] = transformers.__version__
        report["modelopt"] = getattr(modelopt, "__version__", None)
        checks.update(
            python_3_14=sys.version_info[:2] == (3, 14),
            torch_cu132=torch.__version__ == "2.13.0+cu132" and torch.version.cuda == "13.2",
            gpu_available=torch.cuda.is_available(),
            transformers_modelopt_compatible=(5, 9, 0) <= version_tuple(transformers.__version__) < (5, 10, 0),
            modelopt_torch_import=True,
        )
    except Exception as error:  # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        checks["imports"] = False

    report["status"] = "pass" if checks and all(checks.values()) else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
