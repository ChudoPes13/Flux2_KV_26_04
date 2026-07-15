# FLUX.2 Klein-KV на vLLM (RTX 5090, Blackwell, NVFP4)

Native Linux-проект для запуска **FLUX.2 Klein 9B-KV** через **vLLM image-generation backend** на NVIDIA GeForce RTX 5090 (Blackwell, 32 GiB VRAM) с целевой NVFP4-квантизацией. Проект развивается поэтапно: сначала воспроизводимая диагностика окружения и совместимости vLLM + Blackwell + CUDA 13.2, затем BF16 baseline, потом исследовательская NVFP4-квантизация и только после этого — acceptance-сравнение BF16 vs NVFP4.

> Статус: переписан stub под vLLM vision/image-gen backend. Прежний TensorRT-LLM / VisualGen путь намеренно удалён как несоответствующий новой целевой архитектуре.

## Цель ближайшего этапа

Получить на одной и той же машине, на одной и той же модели и при одних и тех же seed/конфигурации:

1. Воспроизводимый BF16 baseline через vLLM (`data/output/bf16_baseline/*.png` + `data/diagnostics/bf16_baseline.json`).
2. Исследовательский NVFP4-quantized checkpoint с явно задокументированным списком квантизируемых модулей и calibration-промптов.
3. Запуск NVFP4 checkpoint через vLLM с теми же промптами и seed (`data/output/nvfp4/*.png` + `data/diagnostics/nvfp4_run.json`).
4. Сравнительный отчёт `data/diagnostics/comparison_bf16_vs_nvfp4.json` (PSNR/SSIM, VRAM, latency, throughput).

Цель **не состоит** в получении изображения любой ценой и **не состоит** в автоматической переквантизации всех слоёв без анализа.

## Документация

- [Инструкции для LLM и участников](agents.md)
- [План работ](plan.md)
- [Журнал изменений](changelog.md)
- [Архитектура и окружение](architecture.md)
- [Полный GPU-first процесс](workflow.md)
- [Правила разработки и GitHub-процесс](rules.md)
- [Воспроизводимая установка рабочей конфигурации](INSTALLATION.md)
- [Запуск и проверки](run.md)

## Структура

```text
configs/        Версионируемая конфигурация проекта (project.yaml)
src/flux2_kv/   Python-пакет проекта
scripts/        Исполняемые диагностические и рабочие сценарии
tests/          Автоматические тесты
data/           Локальные input, cache, diagnostics и output (не в Git)
models/         Локальные веса моделей (не в Git)
.github/        Шаблон pull request (планируется)
```

Подробное описание слоёв исходного кода — в [architecture.md](architecture.md).

## Быстрый старт окружения

На целевой Ubuntu-системе с CUDA Toolkit 13.2 и Python 3.14:

```bash
source scripts/activate_remote.sh
python scripts/00_ubuntu_check.py --output data/diagnostics/ubuntu_env_check.json
python scripts/01_vllm_smoke.py --output data/diagnostics/vllm_smoke.json
```

`activate_remote.sh` ожидает созданный Python-3.14 venv `.venv` с установленным vLLM из исходников под CUDA 13.2. vLLM устанавливается из source-ветки, совместимой с Blackwell SM 12.0 и PyTorch cu132. Подробные требования и проверки — в [INSTALLATION.md](INSTALLATION.md) и [run.md](run.md).

## Два изолированных Python-окружения

- `.venv` — единственное runtime-окружение vLLM, PyTorch cu132, compressed-tensors, huggingface_hub; активируется `source scripts/activate_remote.sh`.
- `.venv-modelopt` — отдельное окружение для NVIDIA ModelOpt (NVFP4 quantization) и Hugging Face CLI; активируется `source scripts/activate_modelopt_remote.sh`.

Оба окружения используют Python 3.14 и PyTorch `2.13.0+cu132`. Нельзя устанавливать ModelOpt-зависимости или `wget` в `.venv` ради удобства: это меняет проверенный vLLM runtime. Подробности — в [INSTALLATION.md](INSTALLATION.md).

## Рабочая машина

Все GPU-задачи выполняются на целевой Ubuntu-машине с NVIDIA GeForce RTX 5090 (Blackwell, compute capability 12.0, 32 GiB VRAM), Python 3.14.4, CUDA Toolkit 13.2.78 и актуальным проприетарным драйвером NVIDIA. Точный host, driver, kernel и free VRAM фиксируются в `data/diagnostics/ubuntu_env_check.json` на каждом запуске — прошлые замеры не считаются вечными. Секреты доступа, токены и приватные пути в репозиторий не записываются.

Полная последовательность от входных файлов до сравнительного отчёта описана в [workflow.md](workflow.md). Под «всё на GPU» проект понимает весь ML inference (text encoding, transformer denoising, VAE decode) и обработку тензоров; файловый I/O, парсинг YAML/JSON и проверка метаданных остаются на CPU.

## Важные ограничения

- Docker не используется.
- Нативный Python проекта — 3.14; CUDA Toolkit 13.2 — единственная разрешённая ветка CUDA.
- Backend — vLLM (`model_type=image`, FLUX.2-dispatch). TensorRT-LLM, VisualGen, Diffusers fallback и ComfyUI запрещены.
- NVFP4 применяется ровно к тем модулям, которые явно перечислены в `configs/project.yaml::quantization.target_modules` после Sprint 004. VAE, tokenizer, T5 text encoder и `final_layer` остаются BF16, если иное не доказано отдельным спринтом.
- BF16 baseline обязан пройти до любой NVFP4 работы. NVFP4 acceptance сравнивается с этим baseline, а не с произвольным изображением.
- OOM — валидный классифицированный результат, а не повод автоматически снизить resolution, precision или переключить runtime.
- Каждый этап выполняется в отдельной ветке `sprint/<номер>-<краткое-имя>` через PR в `main`.
