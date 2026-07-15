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

1. Поместите разрешённые веса модели в `models/` или укажите их абсолютный путь.
2. Выполните отдельную сборку TensorRT engine для этой модели на данной машине.
3. Запустите acceptance-тест модели с небольшим входом и сохраните результат в `data/diagnostics/`.

Не используйте `pip install -U torch`, `pip install tensorrt-llm` или CUDA 13.3: это нарушит зафиксированную рабочую конфигурацию. Подробное восстановление окружения описано в [INSTALLATION.md](INSTALLATION.md).
