"""FLUX.2 Klein-KV native Ubuntu runtime package (vLLM image-generation backend).

This package contains helper modules used by scripts/00..08. The scripts are the
entrypoints; this package provides reusable building blocks:

- config: load and validate configs/project.yaml
- env: native Ubuntu/CUDA/GPU/vLLM diagnostics
- diagnostics: JSON report writers
- vllm_runner: wrapper around vLLM LLM(model_type='image')
- checkpoint_inspection: safetensors and compressed-tensors header checks
- image_io: PNG/JPEG normalization and output helpers
- quantization: ModelOpt NVFP4 pipeline (executed from .venv-modelopt)
- compare: PSNR/SSIM, latency and VRAM diff between runs
- report: aggregate run report

Modules are implemented incrementally across sprints 101..106 (see plan.md).
"""
