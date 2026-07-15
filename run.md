# Запуск проекта

Все команды выполняйте на целевой Ubuntu-машине из корня репозитория (например, `~/Flux2_KV_VLLM`).

## 1. Активировать vLLM runtime окружение

```bash
cd ~/Flux2_KV_VLLM
source scripts/activate_remote.sh
```

Скрипт проверяет CUDA 13.2 и `.venv` на Python 3.14, активирует venv и выставляет `CUDA_HOME`, `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES=0`. vLLM runtime не зависит от UCX/NIXL — эти библиотеки не требуются для single-GPU vLLM inference.

## 2. Проверить стек

```bash
python scripts/00_ubuntu_check.py --strict --output data/diagnostics/ubuntu_env_check.json
python scripts/01_vllm_smoke.py --strict --output data/diagnostics/vllm_smoke.json
pytest -q tests/test_runtime_smoke.py
```

Успешный результат означает, что доступны GPU, PyTorch cu132, vLLM и FLUX.2 model class зарегистрирован в vLLM registry. Эти проверки не скачивают модель и не запускают генерацию.

## 3. Краткий интерактивный smoke-test

```bash
python -c "import torch, vllm; print(torch.__version__, torch.cuda.get_device_name(0), vllm.__version__)"
```

Дополнительно можно проверить, что FLUX.2 зарегистрирован:

```bash
python - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
flux = [a for a in archs if 'flux' in a.lower()]
print('flux archs:', flux)
PY
```

## 4. Перед запуском модели

Сначала убедитесь, что аккаунт Hugging Face авторизован и имеет доступ к лицензированному BFL repository, затем выполните read-only preflight. Он не скачивает весов и показывает точный объём, полученный из метаданных репозитория:

```bash
source scripts/activate_modelopt_remote.sh
hf auth login
source scripts/activate_remote.sh
python scripts/03_model_readiness.py --strict --output data/diagnostics/model_readiness.json
```

Hugging Face CLI намеренно находится в `.venv-modelopt`; token сохраняется в пользовательском Hub cache и затем читается библиотекой `huggingface_hub` из `.venv` без установки CLI в vLLM runtime.

При `"status": "pass"` загрузите BFL assets:

```bash
python scripts/04_download_models.py --output data/diagnostics/model_download.json
```

После `"status": "pass"`:

1. Убедитесь, что `models/bfl/` содержит `model_index.json`, scheduler, tokenizer, transformer, vae, text_encoder.
2. Перейдите к BF16 baseline.

Не используйте `pip install -U torch`, `pip install vllm` из PyPI или CUDA 13.3: это нарушит зафиксированную рабочую конфигурацию. vLLM устанавливается только из source-ветки, зафиксированной в [INSTALLATION.md](INSTALLATION.md).

## 5. BF16 baseline run

```bash
source scripts/activate_remote.sh
python scripts/05_bf16_baseline.py --output data/diagnostics/bf16_baseline.json
```

Скрипт:
- читает `data/input/prompt.txt` (если отсутствует — выводит явную ошибку с инструкцией);
- загружает FLUX.2 Klein-KV через vLLM `LLM(model=..., model_type='image', dtype='bfloat16')`;
- генерирует изображение 1024×1024, steps=4, seed=42, guidance=4.0;
- сохраняет `data/output/bf16_baseline/42.png` и JSON с VRAM before/after, latency, prompt hash.

## 6. NVFP4 quantization (отдельный venv)

```bash
cd ~/Flux2_KV_VLLM
source scripts/activate_modelopt_remote.sh
python scripts/06_quantize_nvfp4.py --output data/diagnostics/quantization.json
```

Ожидаемый результат — `"status": "pass"`, checkpoint в `models/bfl_nvfp4/`, `target_modules` зафиксирован в JSON и `configs/project.yaml`. ModelOpt запускается только из `.venv-modelopt`; не пытайтесь установить ModelOpt в vLLM runtime venv.

## 7. NVFP4 quantized run

```bash
source scripts/activate_remote.sh
python scripts/07_quantized_run.py --output data/diagnostics/nvfp4_run.json
```

vLLM загружает compressed-tensors checkpoint из `models/bfl_nvfp4/` и использует тот же prompt/seed, что и baseline.

## 8. Сравнение и финальный verdict

```bash
python scripts/08_compare_results.py --output data/diagnostics/comparison_bf16_vs_nvfp4.json
```

Сохраняются PSNR/SSIM/latency/VRAM delta и side-by-side PNG в `data/output/comparison/`. Verdict `accept` / `investigate` / `reject` выбирается по порогам из [workflow.md](workflow.md#8-сравнение-bf16-vs-nvfp4).

## 9. Диагностика ModelOpt venv

Если нужно перезапустить только ModelOpt smoke-test:

```bash
cd ~/Flux2_KV_VLLM
source scripts/activate_modelopt_remote.sh
python scripts/02_modelopt_smoke.py --output data/diagnostics/modelopt_smoke.json
```

Ожидаемый результат — `"status": "pass"`, PyTorch `2.13.0+cu132`, CUDA `13.2`, `nvidia-modelopt 0.45.0` и совместимый `transformers`.
