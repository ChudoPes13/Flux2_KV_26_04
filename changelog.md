# Журнал изменений

Этот файл append-only: каждое новое изменение добавляется в **конец** файла отдельной датированной записью. Не редактируйте и не удаляйте исторические записи, кроме исправления фактической опечатки.

## 2026-07-14 — Спринт 000: базовая структура и документация

- Создан новый GitHub-репозиторий с базовой веткой `main`.
- Добавлены `README.md`, `agents.md`, `plan.md`, `changelog.md`, `architecture.md` и `rules.md` на русском языке.
- Зафиксирована целевая native Linux-архитектура: Ubuntu 26.04, CUDA 13.2, Blackwell/RTX 50XX, Python 3.13 и PyTorch cu132.
- Явно исключён Docker; зафиксирован branch → push → PR → merge workflow для каждого последующего спринта.
- Добавлены стартовые каталоги, конфигурация и шаблон PR без переноса непроверенной исторической Windows/Docker-логики.

## 2026-07-14 — Спринт 001: удалённая GPU-площадка и workflow

- Выполнена read-only диагностика удалённой Ubuntu 26.04 машины с RTX 5060 Ti Blackwell (CC 12.0), 16 GiB VRAM, driver 595.71.05 и CUDA runtime 13.2.
- Добавлен `workflow.md` с GPU-first процессом: input, выбор моделей и framework, native environment gate, NVFP4 политика, обработка, диагностические развилки и output.
- Зафиксированы текущие blockers окружения: отсутствуют Python 3.13, CUDA Toolkit/`nvcc`, PyTorch, TensorRT и TensorRT-LLM; системный Python 3.14.4 не используется для проекта.
- Уточнено правило: `full` ApacheOne NVFP4 — первичный вариант; `txtattn_bf16` не является silent fallback.

## 2026-07-15 — Спринт 002: native environment gate (blocked)

- Установлены user-local CPython 3.12.13 и новый project venv; прежний Python-3.13 venv сохранён на удалённой машине как локальный резервный каталог и не добавлялся в Git.
- Установлены и проверены CUDA Toolkit 13.2.78, PyTorch 2.13.0+cu132, TensorRT 11.1.0.106; PyTorch видит RTX 5060 Ti (CC 12.0). CUDA 13.3 не устанавливалась.
- Добавлены `scripts/activate_remote.sh` и `scripts/00_ubuntu_check.py` для активации Python-3.12 venv и сохранения native environment report.
- Resolver TensorRT-LLM 1.2.1 проверен с constraints `torch==2.13.0+cu132`, `torchvision==0.28.0+cu132` и `cuda-python==13.2.0`. Он остановился на жёстком конфликте: TensorRT-LLM требует `torch >=2.9.1, <=2.10.0a0`.
- TensorRT-LLM/VisualGen, Hugging Face login и загрузка моделей намеренно не выполнялись. Спринт остаётся `blocked` до решения о совместимом официальном wheel либо отдельном одобренном source-build.

## 2026-07-15 — Спринт 002: native runtime, ModelOpt и model readiness

- Native source-build TensorRT-LLM `1.3.0rc20` успешно запущен с Python 3.14.4, CUDA Toolkit 13.2.78, `torch 2.13.0+cu132`, TensorRT 11, VisualGen, NIXL и UCX; environment/runtime smoke tests возвращают `pass`.
- Зафиксировано разделение окружений: `.venv` служит только TensorRT-LLM runtime, `.venv-modelopt` — NVIDIA ModelOpt, совместимому Transformers и Hugging Face CLI. Оба используют Python 3.14.
- Добавлены ModelOpt smoke-test, model readiness/download scripts и no-copy VisualGen runtime layout checks.
- Root LVM на NVMe расширен online с 100 GiB до 936 GiB, поэтому выборочная загрузка ApacheOne checkpoints, BFL companion assets и 4-bit text encoder имеет достаточный запас места.

## 2026-07-15 — Sprint 100: rewrite stub под vLLM vision backend

- Проект портирован с TensorRT-LLM VisualGen на **vLLM image-generation backend**; целевая машина изменена с RTX 5060 Ti (16 GiB) на **RTX 5090 (32 GiB VRAM, Blackwell, SM 12.0)**.
- Все MD-документы (`README.md`, `architecture.md`, `agents.md`, `rules.md`, `workflow.md`, `run.md`, `INSTALLATION.md`, `plan.md`) переписаны под vLLM + RTX 5090 + NVFP4 цель. Исторические записи оставлены без изменений для контекста.
- `configs/project.yaml` приведён к новой модели: `backend: vllm`, `gpu_model: RTX 5090`, `vram_gib: 32`, секции `vllm` и `quantization` с явно заданным `ignore_modules: [vae, tokenizer, t5, final_layer]`.
- `pyproject.toml` переименован в `flux2-kv-vllm`, добавлены `scikit-image` (PSNR/SSIM), `compressed-tensors`, `accelerate` в `modelopt` extras.
- Удалены VisualGen-специфичные скрипты `05_prepare_runtime_layout.py` и `06_check_visualgen_layout.py`. Добавлены заглушки `05_bf16_baseline.py`, `06_quantize_nvfp4.py`, `07_quantized_run.py`, `08_compare_results.py`.
- `scripts/01_runtime_smoke.py` переименован в `01_vllm_smoke.py`; `00_ubuntu_check.py` обновлён (импорт vLLM вместо TensorRT-LLM). `activate_remote.sh` упрощён: UCX/NIXL/TRT-LLM lib paths удалены, оставлены только CUDA, PyTorch, vLLM.
- Нумерация спринтов перезапущена с 100 в `plan.md`, чтобы избежать коллизий с историческими спринтами 000–007 TensorRT-LLM era.
- Не включает установку vLLM, загрузку модели или генерацию — это Sprint 101–106.
