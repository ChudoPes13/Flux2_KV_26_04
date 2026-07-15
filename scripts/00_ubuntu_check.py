#!/usr/bin/env python3
"""Write a native Ubuntu/CUDA/vLLM compatibility report without Docker checks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def command_output(*command: str) -> dict[str, Any]:
    """Run a short read-only diagnostic command and retain its output."""
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": list(command)}
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {
        "available": True,
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def optional_module(name: str) -> tuple[dict[str, Any], Any | None]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # Diagnostic reports must include import failures.
        return ({"installed": False, "error": f"{type(exc).__name__}: {exc}"}, None)
    return ({"installed": True, "version": getattr(module, "__version__", None)}, module)


def cuda_toolkit() -> dict[str, Any]:
    result = command_output("nvcc", "--version")
    release = None
    if result.get("returncode") == 0:
        match = re.search(r"release\s+(\d+\.\d+)", result.get("stdout", ""))
        if match:
            release = match.group(1)
    return {"release": release, "cuda_home": os.environ.get("CUDA_HOME"), "probe": result}


def nvidia_gpu() -> dict[str, Any]:
    result = command_output(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    )
    values: dict[str, Any] = {"probe": result}
    if result.get("returncode") != 0 or not result.get("stdout"):
        return values
    fields = [field.strip() for field in result["stdout"].splitlines()[0].split(",")]
    if len(fields) != 5:
        values["parse_error"] = "unexpected nvidia-smi field count"
        return values
    values.update(
        {
            "name": fields[0],
            "driver": fields[1],
            "memory_total_mib": fields[2],
            "memory_free_mib": fields[3],
            "compute_capability": fields[4],
        }
    )
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/ubuntu_env_check.json"),
        help="JSON destination (created if necessary).",
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless every gate passes.")
    args = parser.parse_args()

    operating_system = os_release()
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "platform": {"system": sys.platform, "machine": platform.machine()},
        "os_release": operating_system,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "cuda_toolkit": cuda_toolkit(),
        "gpu": nvidia_gpu(),
        "library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "vllm_attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
    }

    torch_info, torch = optional_module("torch")
    if torch is not None:
        torch_info["cuda"] = getattr(torch.version, "cuda", None)
        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        if torch_info["cuda_available"]:
            torch_info["device_name"] = torch.cuda.get_device_name(0)
            torch_info["capability"] = list(torch.cuda.get_device_capability(0))
    report["pytorch"] = torch_info
    report["vllm"], vllm = optional_module("vllm")
    report["compressed_tensors"], _ = optional_module("compressed_tensors")

    # FLUX.2 model class registry probe — only if vLLM import succeeded.
    flux_archs: list[str] = []
    if vllm is not None:
        try:
            from vllm.model_executor.models.registry import ModelRegistry

            flux_archs = [a for a in ModelRegistry.get_supported_archs() if "flux" in a.lower()]
        except Exception as exc:  # noqa: BLE001
            report["vllm"]["registry_error"] = f"{type(exc).__name__}: {exc}"
    report["vllm"]["flux_archs"] = flux_archs

    capability = torch_info.get("capability", [])
    checks = {
        "ubuntu_26_04": operating_system.get("ID") == "ubuntu"
        and operating_system.get("VERSION_ID") == "26.04",
        "python_3_14": sys.version_info[:2] == (3, 14),
        "cuda_toolkit_13_2": report["cuda_toolkit"]["release"] == "13.2",
        "pytorch_cu132": torch_info.get("version", "").endswith("+cu132")
        and torch_info.get("cuda") == "13.2",
        "pytorch_cuda_available": torch_info.get("cuda_available") is True,
        "blackwell_or_newer": bool(capability) and capability[0] >= 10,
        "vllm_import": report["vllm"].get("installed") is True,
        "compressed_tensors_import": report["compressed_tensors"].get("installed") is True,
        "flux_arch_registered": len(flux_archs) > 0,
    }
    report["checks"] = checks
    report["status"] = "pass" if all(checks.values()) else "blocked"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not args.strict or report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
