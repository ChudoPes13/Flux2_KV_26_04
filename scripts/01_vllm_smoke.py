#!/usr/bin/env python3
"""Validate the installed vLLM image-generation runtime and FLUX.2 model registration.

This script does NOT load model weights or run generation. It only verifies that
vLLM, PyTorch cu132, Compressed-Tensors and the FLUX model class are importable
and that the GPU is a Blackwell-class device compatible with the project.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/vllm_smoke.json"),
        help="JSON destination.",
    )
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a check fails.")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "cuda_home": os.environ.get("CUDA_HOME"),
        "vllm_attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
    }

    torch_info, torch = import_module("torch")
    report["torch"] = torch_info
    if torch is not None:
        torch_info.update(
            {
                "cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "capability": list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None,
            }
        )

    report["vllm"], vllm = import_module("vllm")
    report["compressed_tensors"], _ = import_module("compressed_tensors")
    report["huggingface_hub"], _ = import_module("huggingface_hub")

    # FLUX.2 model class registry probe — confirms vLLM can dispatch FLUX family.
    flux_archs: list[str] = []
    registry_error: str | None = None
    if vllm is not None:
        try:
            from vllm.model_executor.models.registry import ModelRegistry

            flux_archs = [a for a in ModelRegistry.get_supported_archs() if "flux" in a.lower()]
        except Exception as exc:  # noqa: BLE001
            registry_error = f"{type(exc).__name__}: {exc}"
    report["vllm_flux_archs"] = flux_archs
    if registry_error:
        report["vllm_registry_error"] = registry_error

    # Optional: confirm image generation entrypoint is reachable. We do NOT instantiate
    # LLM() here — that is the job of 05_bf16_baseline.py.
    image_gen_entrypoint: dict[str, Any] = {}
    try:
        from vllm import LLM  # noqa: F401

        image_gen_entrypoint["LLM_import"] = True
    except Exception as exc:  # noqa: BLE001
        image_gen_entrypoint["LLM_import"] = False
        image_gen_entrypoint["LLM_import_error"] = f"{type(exc).__name__}: {exc}"
    report["image_gen_entrypoint"] = image_gen_entrypoint

    checks = {
        "python_3_14": sys.version_info[:2] == (3, 14),
        "cuda_home_13_2": report["cuda_home"] == "/usr/local/cuda-13.2",
        "torch_cu132": torch_info.get("version") == "2.13.0+cu132"
        and torch_info.get("cuda") == "13.2",
        "gpu_available": torch_info.get("cuda_available") is True,
        "blackwell_or_newer": bool(torch_info.get("capability")) and torch_info["capability"][0] >= 10,
        "vllm_import": report["vllm"].get("ok") is True,
        "compressed_tensors_import": report["compressed_tensors"].get("ok") is True,
        "huggingface_hub_import": report["huggingface_hub"].get("ok") is True,
        "flux_arch_registered": len(flux_archs) > 0,
        "llm_entrypoint_import": image_gen_entrypoint.get("LLM_import") is True,
    }
    report["checks"] = checks
    report["status"] = "pass" if all(checks.values()) else "failed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not args.strict or report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
