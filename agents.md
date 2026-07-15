# Инструкции проекта для LLM

## Контекст

Проект запускает **FLUX.2 Klein 9B-KV** через **vLLM image-generation backend** на Ubuntu 26.04 + CUDA 13.2 + RTX 5090 (Blackwell, CC 12.0, 32 GiB VRAM) + Python 3.14. Целевая задача — воспроизводимый BF16 baseline и последующая NVFP4-квантизация с метрическим сравнением. Каноническая документация: [README](README.md), [архитектура](architecture.md), [правила](rules.md), [план](plan.md) и [журнал](changelog.md).

Рабочая GPU-машина: Ubuntu 26.04, RTX 5090 (Blackwell, CC 12.0, 32 GiB VRAM), CUDA Toolkit 13.2. Пароли, HF tokens, SSH host keys и иные секреты не записывай в Git, конфиги или диагностику. Перед любой GPU-работой повторно проверь фактический `nvidia-smi` и импорт PyTorch/vLLM.

## Порядок работы

1. Прочитай `README.md`, `architecture.md`, `rules.md`, `plan.md` и актуальный `changelog.md`.
2. Проверь рабочее дерево и текущую ветку; не затирай чужие изменения.
3. До изменения логики запусти или подготовь диагностический шаг, который подтверждает причину проблемы.
4. Делай ровно один самостоятельный спринт в одной ветке `sprint/<номер>-<краткое-имя>` (нумерация с 100 после rewrite под vLLM — см. `plan.md`).
5. Добавь/обнови тесты и диагностические поля, если меняется наблюдаемое поведение.
6. Обнови `plan.md`, затем **добавь** новую запись в конец `changelog.md`; существующие записи не переписывай.
7. Выполни локальную проверку, сделай commit, push, PR и merge в `main` согласно `rules.md`.

## Неподвижные требования

- Только native Ubuntu; Docker запрещён.
- Целевой стек: Ubuntu 26.04, CUDA Toolkit 13.2, RTX 5090 / Blackwell, Python 3.14, PyTorch `2.13.0+cu132`.
- Backend — vLLM (`model_type=image`, FLUX.2 dispatch). TensorRT-LLM, VisualGen, Diffusers fallback и ComfyUI запрещены.
- Используются ровно два Python-окружения:
  - `.venv` для vLLM runtime (`scripts/activate_remote.sh`);
  - `.venv-modelopt` для NVIDIA ModelOpt и Hugging Face CLI (`scripts/activate_modelopt_remote.sh`).
  Не смешивай их пакеты.
- PyTorch устанавливается командой `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`.
- Не устанавливай CUDA 13.3 и не разрешай pip заменить `torch 2.13.0+cu132` ради vLLM. При несовместимости pre-built wheel остановись до загрузки модели и сохрани environment report.
- Каждый inference использует GPU 0; `batch_size=1` для baseline и NVFP4 acceptance. Multi-batch допустим только в Sprint 108.
- CPU допустим только для файлового I/O, токенизации/метаданных и orchestration; нейросетевые вычисления и тензоры не должны переходить на CPU fallback.
- Не добавляй Diffusers fallback, ComfyUI, Telegram, очереди или multi-worker.
- Не запускай NVFP4 acceptance, пока BF16 baseline не зафиксирован и не сохранён.
- Не меняй `target_modules` для NVFP4 без нового спринта с метрическим доказательством.
- Не скрывай исключения: каждый существенный сбой должен сохранять диагностический JSON с traceback и версионным контекстом.

## Режимы запуска

| Режим | Назначение | Обязательное поведение |
| --- | --- | --- |
| `bf16_baseline` | Эталонный baseline, единственный для всех сравнений | `mode=bf16_baseline`, `quantization=none`, `dtype=bfloat16`, seed фиксирован |
| `nvfp4_quantized` | Целевой путь после Sprint 004 | `mode=nvfp4_quantized`, `quantization=nvfp4`, `target_modules=[...]`, `calibration_prompts_hash=...`, тот же seed/prompt, что и baseline |

Если NVFP4 run падает с `unsupported_arch` — это ожидаемое ограничение vLLM/Compressed-Tensors на Blackwell, а не повод переключаться на Diffusers или снижать precision.

## Принцип классификации сбоев

- vLLM не импортируется → исследуй окружение, CUDA, driver, версию vLLM, патчи Python 3.14.
- vLLM импортируется, но FLUX.2 model class не зарегистрирован → проверь ветку vLLM, обнови `INSTALLATION.md`, не переходи к download.
- BF16 baseline падает с OOM → RTX 5090 32 GiB должно хватать; исследуй `gpu_memory_utilization`, batch, enabled compile, оставь OOM как классифицированный результат.
- BF16 baseline проходит, NVFP4 run падает на load → проверь compressed-tensors checkpoint, `target_modules`, `ignore_modules`, версию compressed-tensors в обоих venv.
- NVFP4 run проходит, но метрики сильно хуже baseline → расширь `ignore_modules`, добавь calibration prompts, не откатывайся к BF16 автоматически.

## Что включать в отчёты

Минимум: GPU name и compute capability, объём/свободная VRAM до и после, Ubuntu, driver, CUDA runtime, Python executable/version, PyTorch, vLLM version + commit, Compressed-Tensors version, `LD_LIBRARY_PATH`, режим, путь к модели/checkpoint, факт использования cache, seed, hash промпта, stdout/stderr, классификаторы `oom`/`unsupported_arch`/`missing_model`/`invalid_safetensors`/`vllm_import_failed` и полный traceback.

## Как продолжать работу после этого stub

1. Sprint 100 (этот PR) — только docs и структура. После merge переходи к Sprint 101.
2. Sprint 101 — установи vLLM из source под Python 3.14 + CUDA 13.2 + Blackwell SM 12.0. Зафиксируй точный commit и патчи в `INSTALLATION.md`. Дай `scripts/01_vllm_smoke.py` пройти.
3. Sprint 102 — `scripts/03_model_readiness.py` и `scripts/04_download_models.py`. Только после `pass`.
4. Sprint 103 — реализуй `src/flux2_kv/vllm_runner.py` и `scripts/05_bf16_baseline.py`. Сохрани baseline PNG и JSON.
5. Sprint 104 — ModelOpt NVFP4 quantization в `.venv-modelopt`. Сохрани checkpoint в `models/bfl_nvfp4/` и зафиксируй `target_modules`.
6. Sprint 105 — `scripts/07_quantized_run.py` с тем же seed/prompt.
7. Sprint 106 — `scripts/08_compare_results.py` и финальный verdict.

Каждый спринт — отдельная ветка `sprint/<номер>-<краткое-имя>`, PR в `main`, append-only `changelog.md`.
