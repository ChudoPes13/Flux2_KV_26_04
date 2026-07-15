# План проекта

Статусы: `planned` — запланировано, `in progress` — выполняется, `done` — завершено, `blocked` — требуется внешнее решение.

## Спринт 000 — Базовая структура и документация

- Статус: `done`
- Результат: native Ubuntu-ориентированная структура, правила, архитектурное описание, план и append-only changelog.
- Не включает: перенос либо изменение исторических Windows/Docker-скриптов.

## Спринт 001 — Удалённая GPU-площадка и сквозной workflow

- Статус: `done`
- Подтверждена целевая машина: Ubuntu 26.04 LTS, RTX 5060 Ti Blackwell (CC 12.0), 16 GiB VRAM, driver 595.71.05, CUDA runtime 13.2.
- Зафиксирован GPU-first/NVFP4 процесс от input до отчёта в `workflow.md`.
- Первичный baseline зафиксирован до установки native runtime; исторические blockers устранены в Sprint 002.
- Не включает: установку пакетов, загрузку весов или генерацию.

## Спринт 002 — Native Ubuntu environment gate

- Статус: `done`
- Созданы два Python 3.14 окружения: `.venv` для TensorRT-LLM runtime и `.venv-modelopt` для NVIDIA ModelOpt/Hugging Face CLI.
- В `.venv` проверены CUDA Toolkit 13.2.78, `torch 2.13.0+cu132`, TensorRT Python 11.1.0.106, source-build TensorRT-LLM `1.3.0rc20`, VisualGen, NIXL и UCX; `scripts/00_ubuntu_check.py` и `scripts/01_runtime_smoke.py` возвращают `pass`.
- ModelOpt 0.45.0, Transformers 5.9.0 и Torch cu132 проверены отдельным `scripts/02_modelopt_smoke.py`.
- Root LVM расширен с 100 GiB до 936 GiB; модельный preflight и выборочная загрузка assets доступны через `scripts/03_model_readiness.py` и `scripts/04_download_models.py`.

## Спринт 003 — Перенос CPU-only валидаций

- Статус: `planned`
- Адаптировать только независимые от Docker проверки: runtime layout, safetensors inspection, prompt cache и GenerationInputs mock-test.
- Сохранить проверенные инварианты: `prompt_embeds [1,512,12288] bf16`, `text_ids [1,512,4] int64`.
- Acceptance: проверки проходят без GPU generation и создают диагностические артефакты.

## Спринт 004 — Официальный VisualGen smoke gate

- Статус: `planned`
- Добавить проверку официально поддерживаемой VisualGen-модели на native Ubuntu.
- Не исследовать ApacheOne, пока этот шаг не классифицирован.
- Acceptance: `pass/fail` и категория сбоя в отчёте.

## Спринт 005 — ApacheOne load-only диагностика

- Статус: `planned`
- Проверить раздельно `full` и `txtattn_bf16` runtime layout с ApacheOne safetensors.
- Не менять text encoder cache и не добавлять fallback.
- Acceptance: отдельные отчёты и классификация совместимости loader/layout.

## Спринт 006 — Два режима генерации

- Статус: `planned`
- Реализовать `visualgen_prompt_text` только как smoke-test и `cached_embeddings_strict` как строгий путь.
- Acceptance: итоговый `data/diagnostics/rtx50_first_run_report.json` содержит пять контрольных результатов, описанных в README.

## Спринт 007 — Нижнеуровневый adapter (условный)

- Статус: `planned`, запускается только если public VisualGen не принимает external embeddings.
- Спроектировать adapter после сохранённой диагностики; не делать автоматический fallback на prompt text.
