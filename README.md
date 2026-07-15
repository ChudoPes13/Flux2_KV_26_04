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
- [Полный GPU-first процесс](workflow.md)
- [Правила разработки и GitHub-процесс](rules.md)
- [Воспроизводимая установка рабочей конфигурации](INSTALLATION.md)
- [Запуск и проверки](run.md)

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

На целевой Ubuntu-системе с CUDA Toolkit 13.2 и Python 3.14:

```bash
source scripts/activate_remote.sh
python scripts/00_ubuntu_check.py --output data/diagnostics/ubuntu_env_check.json
```

`activate_remote.sh` ожидает созданный Python-3.14 venv. В нём PyTorch всегда ставится только командой `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132`; CUDA 13.3 не используется. Подробные требования и проверки — в [INSTALLATION.md](INSTALLATION.md) и [run.md](run.md).

## Два изолированных Python-окружения

- `.venv` — единственное runtime-окружение TensorRT, TensorRT-LLM, VisualGen, NIXL и UCX; активируется `source scripts/activate_remote.sh`.
- `.venv-modelopt` — отдельное окружение NVIDIA ModelOpt и Hugging Face CLI; активируется `source scripts/activate_modelopt_remote.sh`.

Оба окружения используют Python 3.14. Нельзя устанавливать ModelOpt-зависимости, `wget` или Hugging Face CLI в `.venv` ради удобства: это меняет проверенный TensorRT-LLM runtime.

## Рабочая машина

Все GPU-задачи выполняются на удалённой Ubuntu-машине `192.168.0.206` под пользователем `master`. На момент environment gate это Ubuntu 26.04 LTS с NVIDIA GeForce RTX 5060 Ti (Blackwell, compute capability 12.0), 16 GiB VRAM, драйвером 595.71.05 и CUDA runtime/toolkit 13.2. Секреты доступа, токены и приватные пути в репозиторий не записываются.

Полная последовательность от входных файлов до диагностического отчёта описана в [workflow.md](workflow.md). Под «всё на GPU» проект понимает весь ML inference и обработку тензоров; файловый ввод-вывод, парсинг конфигурации и проверка метаданных остаются на CPU, так как не являются GPU-вычислениями.

## Важные ограничения

- Docker не используется.
- Нативный Python проекта — 3.14; CUDA Toolkit 13.2 — единственная разрешённая ветка CUDA.
- На 2026-07-15 нативная compatibility-сборка TensorRT-LLM `1.3.0rc20` успешно проходит environment и VisualGen smoke-тесты с обязательным `torch 2.13.0+cu132`, NIXL и UCX. Сборка не является официально поддерживаемой NVIDIA комбинацией; детали патчей и ограничения приведены в [INSTALLATION.md](INSTALLATION.md).
- `cached_embeddings_strict` никогда не подменяется prompt-текстом или Diffusers.
- Результат на GPU до Blackwell не является NVFP4 acceptance.
- Каждый этап выполняется в отдельной ветке через PR в `main`.
