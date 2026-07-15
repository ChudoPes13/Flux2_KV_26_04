#!/usr/bin/env python3
"""Validate a prepared checkpoint layout without loading model weights onto the GPU."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("full", "txtattn_bf16"), required=True)
    parser.add_argument(
        "--runtime-root", type=Path, default=Path("data/cache/visualgen_runtime")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/diagnostics/visualgen_layout.json")
    )
    args = parser.parse_args()
    runtime = (args.runtime_root / args.variant).resolve()
    expected = (
        "model_index.json",
        "scheduler",
        "tokenizer",
        "vae",
        "text_encoder",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "variant": args.variant,
        "runtime": str(runtime),
        "checks": {item: (runtime / item).exists() for item in expected},
    }
    try:
        from tensorrt_llm._torch.visual_gen.pipeline_registry import AutoPipeline

        report["detected_pipeline"] = AutoPipeline._detect_from_checkpoint(str(runtime))
        report["checks"]["autopipeline_detection"] = report["detected_pipeline"] == "Flux2Pipeline"  # type: ignore[index]
    except Exception as error:  # noqa: BLE001
        report["error"] = f"{type(error).__name__}: {error}"
        report["checks"]["autopipeline_detection"] = False  # type: ignore[index]
    report["status"] = "pass" if all(report["checks"].values()) else "failed"  # type: ignore[index]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
