#!/usr/bin/env python3
"""Download only the FLUX.2 Klein-KV assets used by this native project.

The BFL repository contains full BF16 transformer and text-encoder weights that
are deliberately excluded: ApacheOne supplies the transformer and the separate
4-bit repository supplies the text encoder.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

COMPONENTS = {
    "apacheone": {
        "repo_id": "ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed",
        "destination": Path("apacheone"),
        "allow_patterns": [
            "flux2-klein-9b-kv-nvfp4.safetensors",
            "flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors",
        ],
    },
    "bfl_companion": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9b-kv",
        "destination": Path("bfl"),
        "allow_patterns": [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "transformer/config.json",
            "vae/*",
        ],
    },
    "text_encoder_4bit": {
        "repo_id": "aifeifei798/FLUX.2-klein-9B-text_encoder-4bit",
        "destination": Path("experimental/text_encoder/aifeifei_4bit"),
        "allow_patterns": ["*"],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--component", choices=[*COMPONENTS, "all"], default="all", help="Asset group to fetch."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("data/diagnostics/model_download.json")
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    selected = COMPONENTS if args.component == "all" else {args.component: COMPONENTS[args.component]}
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "models_root": str(args.models_root.resolve()),
        "dry_run": args.dry_run,
        "components": [],
    }
    try:
        HfApi(token=True).whoami()
        for name, component in selected.items():
            target = args.models_root / component["destination"]
            entry: dict[str, object] = {
                "name": name,
                "repo_id": component["repo_id"],
                "target": str(target.resolve()),
                "allow_patterns": component["allow_patterns"],
            }
            if not args.dry_run:
                snapshot_download(
                    repo_id=component["repo_id"],
                    local_dir=target,
                    allow_patterns=component["allow_patterns"],
                    token=True,
                    max_workers=4,
                )
                files = sorted(path for path in target.rglob("*") if path.is_file())
                entry["downloaded_files"] = len(files)
                entry["downloaded_bytes"] = sum(path.stat().st_size for path in files)
            report["components"].append(entry)  # type: ignore[index]
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
