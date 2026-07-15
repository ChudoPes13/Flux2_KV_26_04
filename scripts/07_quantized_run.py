#!/usr/bin/env python3
"""Sprint 105: NVFP4 quantized run for FLUX.2 Klein-KV via vLLM.

Run this from .venv (vLLM runtime), NOT from .venv-modelopt:
    source scripts/activate_remote.sh
    python scripts/07_quantized_run.py

This is a stub. Sprint 105 implements the actual logic following the contract
in architecture.md and workflow.md.

Contract:
- Read prompt from data/input/prompt.txt (same as bf16_baseline).
- Load compressed-tensors NVFP4 checkpoint from models/bfl_nvfp4/ via vLLM.
- Use the same seed (42), steps (4), guidance (4.0), and 1024x1024 as baseline.
- Save data/output/nvfp4/<seed>.png.
- Save JSON report with: mode=nvfp4_quantized, quantization=nvfp4,
  target_modules, ignore_modules, calibration_prompts_sha256, seed,
  prompt_sha256, vram_before_mib, vram_after_mib, latency_s, status=pass.
- On unsupported_arch / compressed_tensors_load_failed: keep the explicit
  classification in the JSON, do NOT fall back to BF16 or Diffusers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/diagnostics/nvfp4_run.json")
    )
    parser.add_argument("--prompt", type=Path, default=Path("data/input/prompt.txt"))
    parser.add_argument("--nvfp4-model", type=Path, default=Path("models/bfl_nvfp4"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompt_hash: str | None = None
    prompt_error: str | None = None
    if args.prompt.is_file():
        prompt_hash = hashlib.sha256(args.prompt.read_bytes()).hexdigest()
    else:
        prompt_error = f"prompt file not found: {args.prompt}"

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "nvfp4_quantized",
        "quantization": "nvfp4",
        "nvfp4_model_path": str(args.nvfp4_model.resolve()),
        "seed": args.seed,
        "prompt_path": str(args.prompt),
        "prompt_sha256": prompt_hash,
        "prompt_error": prompt_error,
        "target_modules": None,  # read from configs/project.yaml in Sprint 105
        "ignore_modules": ["vae", "tokenizer", "t5", "final_layer"],
        "calibration_prompts_sha256": None,  # copied from quantization.json
        "vram_before_mib": None,
        "vram_after_mib": None,
        "latency_s": None,
        "status": "not_implemented",
        "todo": [
            "Read configs/project.yaml::quantization.target_modules.",
            "Read calibration_prompts_sha256 from data/diagnostics/quantization.json.",
            "Load FLUX.2 via vLLM with quantization=nvfp4 from models/bfl_nvfp4/.",
            "Generate image with same seed/steps/guidance as 05_bf16_baseline.py.",
            "Save data/output/nvfp4/<seed>.png.",
            "Record VRAM before/after and latency.",
            "On failure: classify as unsupported_arch / compressed_tensors_load_failed / oom.",
            "Set status=pass when image is saved and JSON has all required fields.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
