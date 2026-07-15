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
- Обнаружены baseline blockers: Python 3.13, CUDA Toolkit (`nvcc`), PyTorch, TensorRT и TensorRT-LLM ещё не установлены; системный Python — 3.14.4.
- Не включает: установку пакетов, загрузку весов или генерацию.

## Спринт 002 — Native Ubuntu environment gate

- Статус: `blocked`
- Добавлен `scripts/00_ubuntu_check.py`: он проверяет `/etc/os-release`, NVIDIA driver, CUDA Toolkit 13.2, Python 3.12, PyTorch cu132, TensorRT, TensorRT-LLM, GPU capability, VRAM и библиотечные пути; создаёт машиночитаемый JSON без Docker-проверок.
- Подтверждены Python 3.12.13, CUDA Toolkit 13.2.78, PyTorch 2.13.0+cu132 с доступной RTX 5060 Ti и TensorRT 11.1.0.106.
- Blocker: TensorRT-LLM 1.2.1 pre-built wheel для Python 3.12 требует `torch >=2.9.1, <=2.10.0a0`; это конфликтует с обязательным PyTorch cu132 версии 2.13.0. Constraint `cuda-python==13.2.0` не снимает torch-конфликт; CUDA 13.3, понижение PyTorch и source-build не разрешены.
- Acceptance: отчёт правильно возвращает `blocked` и объясняет incompatibility; TensorRT-LLM VisualGen, Hugging Face login и модели не запускаются до внешнего решения.

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
