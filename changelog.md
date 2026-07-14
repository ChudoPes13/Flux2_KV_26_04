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
