# Запуск проекта

Все команды выполняйте на целевой Ubuntu-машине из `~/Flux2_KV_26_04`.

## 1. Активировать окружение

```bash
cd ~/Flux2_KV_26_04
source scripts/activate_remote.sh
```

Скрипт проверяет CUDA 13.2 и `.venv` на Python 3.14, а также устанавливает необходимые пути для CUDA, NCCL, UCX и NIXL.

## 2. Проверить стек

```bash
python scripts/00_ubuntu_check.py --strict --output data/diagnostics/ubuntu_env_check.json
python scripts/01_runtime_smoke.py --strict --output data/diagnostics/runtime_smoke.json
pytest -q tests/test_runtime_smoke.py
```

Успешный результат означает, что доступны GPU, PyTorch cu132, TensorRT, TensorRT-LLM, VisualGen и NIXL transfer binding. Эти проверки не скачивают модель и не запускают генерацию.

## 3. Краткий интерактивный smoke-test

```bash
python -c "import torch, tensorrt_llm, tensorrt_llm.visual_gen; print(torch.__version__, torch.cuda.get_device_name(0), tensorrt_llm.__version__)"
```

## 4. Перед запуском модели

Сначала убедитесь, что аккаунт Hugging Face авторизован и имеет доступ к лицензированному BFL repository, затем выполните read-only preflight. Он не скачивает весов и показывает точный объём, полученный из метаданных репозиториев:

```bash
source scripts/activate_modelopt_remote.sh
hf auth login
source scripts/activate_remote.sh
python scripts/03_model_readiness.py --strict --output data/diagnostics/model_readiness.json
```

Hugging Face CLI намеренно находится в `.venv-modelopt`; token сохраняется в пользовательском Hub cache и затем читается библиотекой `huggingface_hub` из `.venv` без установки CLI в TensorRT-LLM runtime.

При `"status": "pass"` загрузите только нужные assets. Скрипт не скачивает полные BF16 transformer/text-encoder weights из BFL, потому что используются ApacheOne NVFP4 transformer и отдельный 4-bit text encoder:

```bash
python scripts/04_download_models.py --output data/diagnostics/model_download.json
python scripts/05_prepare_runtime_layout.py --variant full
python scripts/05_prepare_runtime_layout.py --variant txtattn_bf16
python scripts/06_check_visualgen_layout.py --variant full
python scripts/06_check_visualgen_layout.py --variant txtattn_bf16
```

После `"status": "pass"`:

1. Поместите разрешённые веса модели в `models/` или укажите их абсолютный путь.
2. Выполните отдельную сборку TensorRT engine для этой модели на данной машине.
3. Запустите acceptance-тест модели с небольшим входом и сохраните результат в `data/diagnostics/`.

Не используйте `pip install -U torch`, `pip install tensorrt-llm` или CUDA 13.3: это нарушит зафиксированную рабочую конфигурацию. Подробное восстановление окружения описано в [INSTALLATION.md](INSTALLATION.md).

## 5. Запуск NVIDIA ModelOpt в отдельном venv

ModelOpt запускается из изолированного `.venv-modelopt`, а не из runtime-окружения TensorRT-LLM:

```bash
cd ~/Flux2_KV_26_04
source scripts/activate_modelopt_remote.sh
python scripts/02_modelopt_smoke.py --output data/diagnostics/modelopt_smoke.json
```

Ожидаемый результат — `"status": "pass"`, PyTorch `2.13.0+cu132`, CUDA `13.2`, `nvidia-modelopt 0.45.0` и Transformers `5.9.0`.
