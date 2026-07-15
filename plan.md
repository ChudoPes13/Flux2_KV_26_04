# План проекта

Статусы: `planned` — запланировано, `in progress` — выполняется, `done` — завершено, `blocked` — требуется внешнее решение.

> Исторические спринты 000–007 (TensorRT-LLM / VisualGen era) сохранены в `changelog.md` для контекста. После rewrite под vLLM нумерация спринтов перезапускается с 100, чтобы избежать коллизий с прежним планом.

## Sprint 100 — Rewrite stub under vLLM vision backend

- Статус: `in progress`
- Результат: все MD-документы переписаны под vLLM image-gen backend + RTX 5090 + NVFP4 цель; `configs/project.yaml` и `pyproject.toml` приведены в соответствие; скрипты `05_*/06_*` VisualGen-layout удалены, добавлены заглушки `05_bf16_baseline.py`, `06_quantize_nvfp4.py`, `07_quantized_run.py`, `08_compare_results.py`.
- Не включает: установку vLLM, загрузку модели, генерацию.

## Sprint 101 — Native Ubuntu environment gate (vLLM edition)

- Статус: `planned`
- Цель: получить `pass` от `scripts/00_ubuntu_check.py --strict` и `scripts/01_vllm_smoke.py --strict` на RTX 5090.
- Включает: установку vLLM из source под Python 3.14 + CUDA 13.2 + Blackwell SM 12.0; фиксацию commit vLLM и патчей в `INSTALLATION.md`.
- Acceptance: `data/diagnostics/ubuntu_env_check.json` и `data/diagnostics/vllm_smoke.json` содержат `"status": "pass"`; vLLM видит RTX 5090 и capability 12.0; FLUX.2 model class зарегистрирован в `vllm.model_executor.models.registry`.

## Sprint 102 — Model readiness и BFL download

- Статус: `planned`
- Цель: read-only preflight + snapshot download `black-forest-labs/FLUX.2-klein-9b-kv` в `models/bfl/`.
- Включает: проверку HF access, disk space, GPU VRAM; `scripts/03_model_readiness.py --strict` и `scripts/04_download_models.py`.
- Acceptance: `data/diagnostics/model_readiness.json` и `data/diagnostics/model_download.json` равны `pass`; в `models/bfl/` присутствуют `model_index.json`, scheduler, tokenizer, transformer, vae, text_encoder.

## Sprint 103 — BF16 baseline run

- Статус: `planned`
- Цель: один детерминированный прогон FLUX.2 Klein-KV через vLLM в BF16.
- Включает: реализацию `src/flux2_kv/vllm_runner.py` и `scripts/05_bf16_baseline.py`; сохранение `data/output/bf16_baseline/<seed>.png` и `data/diagnostics/bf16_baseline.json`.
- Acceptance: PNG сохранён, `mode=bf16_baseline`, `quantization=none`, VRAM before/after зафиксированы, seed=42, prompt hash совпадает с `data/input/prompt.txt`. Это единственный baseline для всех будущих сравнений.

## Sprint 104 — NVFP4 quantization research

- Статус: `planned`
- Цель: определить и задокументировать список модулей для NVFP4 квантизации.
- Включает:
  1. Inspect архитектуры FLUX.2 Klein transformer; собрать список всех `nn.Linear` с их ролями (q/k/v proj, mlp, modulator, etc.).
  2. Зафиксировать calibration prompts в `data/input/calibration_prompts.txt` (≥8 промптов, разнообразная семантика).
  3. Запустить ModelOpt NVFP4 quantization в `.venv-modelopt` с текущим `ignore_modules` (`vae`, `tokenizer`, `t5`, `final_layer`).
  4. Зафиксировать в `architecture.md` и `configs/project.yaml::quantization.target_modules` финальный allowlist.
  5. Сохранить compressed-tensors checkpoint в `models/bfl_nvfp4/`.
- Acceptance: `data/diagnostics/quantization.json` содержит `target_modules`, `ignore_modules`, `calibration_prompts_hash`, размер checkpoint и `status=pass`.

## Sprint 105 — NVFP4 quantized run

- Статус: `planned`
- Цель: прогон NVFP4 checkpoint через vLLM с тем же prompt/seed, что и baseline.
- Включает: реализацию `scripts/07_quantized_run.py`; загрузка checkpoint через compressed-tensors loader vLLM.
- Acceptance: `data/diagnostics/nvfp4_run.json` содержит `mode=nvfp4_quantized`, `quantization=nvfp4`, путь к checkpoint, VRAM before/after, seed, prompt hash, `status=pass`. PNG сохранён в `data/output/nvfp4/<seed>.png`.

## Sprint 106 — BF16 vs NVFP4 comparison и acceptance

- Статус: `planned`
- Цель: метрическое и визуальное сравнение baseline и NVFP4.
- Включает: реализацию `src/flux2_kv/compare.py` и `scripts/08_compare_results.py`; PSNR, SSIM, latency, VRAM delta.
- Acceptance: `data/diagnostics/comparison_bf16_vs_nvfp4.json` содержит все метрики, side-by-side PNG в `data/output/comparison/`, и явный verdict `accept` / `investigate` / `reject`. При `reject` план содержит причину и следующий шаг (расширение ignore_modules, дополнительная calibration, etc.).

## Sprint 107 — (опционально) KV cache quantization

- Статус: `planned`, запускается только после Sprint 106.
- Цель: исследовать FP8/INT8 KV cache через vLLM `kv_cache_dtype` при сохранении NVFP4 weights.
- Acceptance: отдельный отчёт `data/diagnostics/kv_cache_quant.json`; сравнение с baseline и с NVFP4 weights-only.

## Sprint 108 — (опционально) Multi-image batch и throughput

- Статус: `planned`, запускается только после Sprint 106.
- Цель: использовать 32 GiB VRAM headroom для `batch_size>1` и зафиксировать throughput.
- Acceptance: `data/diagnostics/throughput.json` с p50/p95 latency для batch 1/2/4.
