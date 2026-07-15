#!/usr/bin/env python3
"""Sprint 104: NVFP4 quantization for FLUX.2 Klein-KV via NVIDIA ModelOpt.

Run this from .venv-modelopt, NOT from the vLLM runtime venv:
    source scripts/activate_modelopt_remote.sh
    python scripts/06_quantize_nvfp4.py

This is a stub. Sprint 104 implements the actual logic following the contract
in architecture.md and workflow.md.

Contract:
- Load BF16 FLUX.2 transformer from models/bfl/.
- Inspect all nn.Linear modules; classify by role (q/k/v proj, mlp, modulator, etc.).
- Build target_modules allowlist; ignore_modules = configs/project.yaml::quantization.ignore_modules.
- Read calibration prompts from data/input/calibration_prompts.txt (>=8 prompts).
- Run ModelOpt NVFP4 quantization.
- Save compressed-tensors checkpoint to models/bfl_nvfp4/.
- Update configs/project.yaml::quantization.target_modules with the final allowlist.
- Save data/diagnostics/quantization.json with target_modules, ignore_modules,
  calibration_prompts_sha256, checkpoint_bytes, status=pass.
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
        "--output", type=Path, default=Path("data/diagnostics/quantization.json")
    )
    parser.add_argument(
        "--calibration-prompts",
        type=Path,
        default=Path("data/input/calibration_prompts.txt"),
    )
    parser.add_argument("--bf16-model", type=Path, default=Path("models/bfl"))
    parser.add_argument("--nvfp4-output", type=Path, default=Path("models/bfl_nvfp4"))
    args = parser.parse_args()

    calibration_hash: str | None = None
    calibration_error: str | None = None
    if args.calibration_prompts.is_file():
        calibration_hash = hashlib.sha256(args.calibration_prompts.read_bytes()).hexdigest()
    else:
        calibration_error = f"calibration prompts file not found: {args.calibration_prompts}"

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "quantization",
        "scheme": "nvfp4",
        "bf16_model_path": str(args.bf16_model.resolve()),
        "nvfp4_output_path": str(args.nvfp4_output.resolve()),
        "calibration_prompts_path": str(args.calibration_prompts),
        "calibration_prompts_sha256": calibration_hash,
        "calibration_error": calibration_error,
        "target_modules": None,  # filled in Sprint 104
        "ignore_modules": ["vae", "tokenizer", "t5", "final_layer"],
        "checkpoint_bytes": None,
        "status": "not_implemented",
        "todo": [
            "Inspect FLUX.2 transformer architecture; classify all nn.Linear by role.",
            "Decide target_modules allowlist (default: attention projections + MLP; skip modulators).",
            "Run ModelOpt NVFP4 quantization with calibration prompts.",
            "Save compressed-tensors checkpoint to models/bfl_nvfp4/.",
            "Update configs/project.yaml::quantization.target_modules.",
            "Set status=pass when checkpoint is saved and target_modules is documented.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
