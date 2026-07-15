#!/usr/bin/env python3
"""Create a no-copy FLUX.2 Klein VisualGen layout from downloaded model assets."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

APACHEONE_FILES = {
    "full": "flux2-klein-9b-kv-nvfp4.safetensors",
    "txtattn_bf16": "flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors",
}


def link_once(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"Existing link points elsewhere: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace a non-link: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=APACHEONE_FILES, required=True)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--runtime-root", type=Path, default=Path("data/cache/visualgen_runtime")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/diagnostics/runtime_layout.json")
    )
    args = parser.parse_args()

    models = args.models_root.resolve()
    runtime = (args.runtime_root / args.variant).resolve()
    transformer = runtime / "transformer"
    links = {
        runtime / "model_index.json": models / "bfl/model_index.json",
        runtime / "scheduler": models / "bfl/scheduler",
        runtime / "tokenizer": models / "bfl/tokenizer",
        runtime / "vae": models / "bfl/vae",
        runtime / "text_encoder": models / "experimental/text_encoder/aifeifei_4bit",
        transformer / "config.json": models / "bfl/transformer/config.json",
        transformer / "diffusion_pytorch_model.safetensors": models
        / "apacheone"
        / APACHEONE_FILES[args.variant],
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "variant": args.variant,
        "runtime": str(runtime),
        "links": {},
    }
    try:
        for link, target in links.items():
            link_once(link, target)
            report["links"][str(link.relative_to(runtime))] = str(link.resolve())  # type: ignore[index]
        report["status"] = "pass"
    except Exception as error:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
