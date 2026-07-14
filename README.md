# FLUX.2 Klein-KV на Ubuntu 26.04

Native Linux-проект для диагностического запуска **FLUX.2 Klein 9B-KV NVFP4** через TensorRT-LLM на GPU класса NVIDIA Blackwell / RTX 50XX. Проект развивается поэтапно: сначала воспроизводимая диагностика окружения и совместимости, затем загрузка моделей и только после этого генерация.

> Статус: создана стартовая структура. Рабочие скрипты из исторического Windows/Docker-проекта намеренно не перенесены без отдельной адаптации и проверки на Ubuntu.

## Цель ближайшего этапа

Получить классифицированный отчёт о пяти проверках: официальная модель VisualGen, загрузка ApacheOne `full`, загрузка `txtattn_bf16`, prompt-text smoke-test и строгий режим cached embeddings. Цель не состоит в получении изображения любой ценой.

## Документация

- [Инструкции для LLM и участников](agents.md)
- [План работ](plan.md)
- [Журнал изменений](changelog.md)
- [Архитектура и окружение](architecture.md)
- [Правила разработки и GitHub-процесс](rules.md)

## Структура

```text
configs/        Версионируемая конфигурация проекта
src/flux2_kv/   Python-пакет проекта
scripts/        Исполняемые диагностические и рабочие сценарии
tests/          Автоматические тесты
data/           Локальные input, cache, diagnostics и output (не в Git)
models/         Локальные веса моделей (не в Git)
.github/        Шаблон pull request
```

## Быстрый старт окружения

На целевой Ubuntu-системе с CUDA 13.2:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e '.[dev]'
```

Установка TensorRT и TensorRT-LLM выполняется отдельно по актуальной официальной инструкции NVIDIA, совместимой с фактическими драйвером, CUDA 13.2 и Blackwell GPU. Подробные требования и проверки — в [architecture.md](architecture.md).

## Важные ограничения

- Docker не используется.
- `cached_embeddings_strict` никогда не подменяется prompt-текстом или Diffusers.
- Результат на GPU до Blackwell не является NVFP4 acceptance.
- Каждый этап выполняется в отдельной ветке через PR в `main`.
