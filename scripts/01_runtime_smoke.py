#!/usr/bin/env python3
"""Validate the installed native TensorRT-LLM, VisualGen, UCX, and NIXL runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def import_module(name: str) -> tuple[dict[str, Any], Any | None]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None
    return {"ok": True, "version": getattr(module, "__version__", None)}, module


def unresolved_dependencies(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["ldd", str(path)], check=False, capture_output=True, text=True, timeout=20
    )
    lines = (completed.stdout + completed.stderr).splitlines()
    return {
        "ok": completed.returncode == 0 and not any("not found" in line for line in lines),
        "path": str(path),
        "unresolved": [line.strip() for line in lines if "not found" in line],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/runtime_smoke.json"),
        help="JSON destination.",
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a check fails.")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "cuda_home": os.environ.get("CUDA_HOME"),
    }
    torch_info, torch = import_module("torch")
    report["torch"] = torch_info
    if torch is not None:
        torch_info.update(
            {
                "cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    report["tensorrt"], _ = import_module("tensorrt")
    report["tensorrt_llm"], tensorrt_llm = import_module("tensorrt_llm")
    report["visual_gen"], _ = import_module("tensorrt_llm.visual_gen")
    report["nixl_transfer_binding"], _ = import_module(
        "tensorrt_llm.tensorrt_llm_transfer_agent_binding"
    )

    libraries: dict[str, Any] = {}
    if tensorrt_llm is not None:
        libs = Path(tensorrt_llm.__file__).parent / "libs"
        for name in ("libtensorrt_llm_nixl_wrapper.so", "libtensorrt_llm_ucx_wrapper.so"):
            path = libs / name
            libraries[name] = (
                unresolved_dependencies(path) if path.is_file() else {"ok": False, "missing": str(path)}
            )
    report["dynamic_libraries"] = libraries

    checks = {
        "python_3_14": sys.version_info[:2] == (3, 14),
        "cuda_home_13_2": report["cuda_home"] == "/usr/local/cuda-13.2",
        "torch_cu132": torch_info.get("version") == "2.13.0+cu132"
        and torch_info.get("cuda") == "13.2",
        "gpu_available": torch_info.get("cuda_available") is True,
        "tensorrt_import": report["tensorrt"].get("ok") is True,
        "tensorrt_llm_import": report["tensorrt_llm"].get("ok") is True,
        "visual_gen_import": report["visual_gen"].get("ok") is True,
        "nixl_transfer_binding_import": report["nixl_transfer_binding"].get("ok") is True,
        "nixl_wrapper_resolved": libraries.get("libtensorrt_llm_nixl_wrapper.so", {}).get("ok")
        is True,
        "ucx_wrapper_resolved": libraries.get("libtensorrt_llm_ucx_wrapper.so", {}).get("ok")
        is True,
    }
    report["checks"] = checks
    report["status"] = "pass" if all(checks.values()) else "failed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not args.strict or report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
