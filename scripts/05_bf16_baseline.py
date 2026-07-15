#!/usr/bin/env python3
"""Sprint 103: BF16 baseline run for FLUX.2 Klein-KV via vLLM image generation.

This is a stub. Sprint 103 implements the actual logic following the contract
in architecture.md and workflow.md.

Contract:
- Read prompt from data/input/prompt.txt (fail with explicit diagnostic if missing).
- Load FLUX.2 Klein-KV via vLLM LLM(model=..., model_type='image', dtype='bfloat16').
- Generate 1024x1024 image, steps=4, seed=42, guidance=4.0 (see configs/project.yaml).
- Save data/output/bf16_baseline/<seed>.png.
- Save JSON report with: mode=bf16_baseline, quantization=none, dtype=bfloat16,
  seed, prompt_sha256, vram_before_mib, vram_after_mib, latency_s, status=pass.

Until Sprint 103 is done, this script writes a `not_implemented` diagnostic JSON
so the workflow gate fails loudly rather than silently skipping the baseline.
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
        "--output", type=Path, default=Path("data/diagnostics/bf16_baseline.json")
    )
    parser.add_argument("--prompt", type=Path, default=Path("data/input/prompt.txt"))
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
        "mode": "bf16_baseline",
        "quantization": "none",
        "dtype": "bfloat16",
        "seed": args.seed,
        "prompt_path": str(args.prompt),
        "prompt_sha256": prompt_hash,
        "prompt_error": prompt_error,
        "status": "not_implemented",
        "todo": [
            "Implement src/flux2_kv/vllm_runner.py: load FLUX.2 via vLLM image-gen entrypoint.",
            "Generate image at configs/project.yaml::generation settings.",
            "Save data/output/bf16_baseline/<seed>.png.",
            "Record VRAM before/after (torch.cuda.mem_get_info) and latency.",
            "Set status=pass when image is saved and JSON has all required fields.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
