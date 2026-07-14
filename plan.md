# План проекта

Статусы: `planned` — запланировано, `in progress` — выполняется, `done` — завершено, `blocked` — требуется внешнее решение.

## Спринт 000 — Базовая структура и документация

- Статус: `done`
- Результат: native Ubuntu-ориентированная структура, правила, архитектурное описание, план и append-only changelog.
- Не включает: перенос либо изменение исторических Windows/Docker-скриптов.

## Спринт 001 — Native Ubuntu environment gate

- Статус: `planned`
- Добавить `scripts/00_ubuntu_check.py` и модуль диагностики native среды.
- Проверять `/etc/os-release`, NVIDIA driver, CUDA 13.2 runtime, Python 3.13, PyTorch cu132, TensorRT, TensorRT-LLM, GPU capability, VRAM и библиотечные пути.
- Сформировать машиночитаемый JSON-отчёт; Docker-проверки не добавлять.
- Acceptance: отчёт чётко определяет, пригодна ли машина для NVFP4 Blackwell-проверки, и объясняет причину отказа.

## Спринт 002 — Перенос CPU-only валидаций

- Статус: `planned`
- Адаптировать только независимые от Docker проверки: runtime layout, safetensors inspection, prompt cache и GenerationInputs mock-test.
- Сохранить проверенные инварианты: `prompt_embeds [1,512,12288] bf16`, `text_ids [1,512,4] int64`.
- Acceptance: проверки проходят без GPU generation и создают диагностические артефакты.

## Спринт 003 — Официальный VisualGen smoke gate

- Статус: `planned`
- Добавить проверку официально поддерживаемой VisualGen-модели на native Ubuntu.
- Не исследовать ApacheOne, пока этот шаг не классифицирован.
- Acceptance: `pass/fail` и категория сбоя в отчёте.

## Спринт 004 — ApacheOne load-only диагностика

- Статус: `planned`
- Проверить раздельно `full` и `txtattn_bf16` runtime layout с ApacheOne safetensors.
- Не менять text encoder cache и не добавлять fallback.
- Acceptance: отдельные отчёты и классификация совместимости loader/layout.

## Спринт 005 — Два режима генерации

- Статус: `planned`
- Реализовать `visualgen_prompt_text` только как smoke-test и `cached_embeddings_strict` как строгий путь.
- Acceptance: итоговый `data/diagnostics/rtx50_first_run_report.json` содержит пять контрольных результатов, описанных в README.

## Спринт 006 — Нижнеуровневый adapter (условный)

- Статус: `planned`, запускается только если public VisualGen не принимает external embeddings.
- Спроектировать adapter после сохранённой диагностики; не делать автоматический fallback на prompt text.
