# Инструкции проекта для LLM

## Контекст

Проект запускает FLUX.2 Klein 9B-KV NVFP4 на **Ubuntu 26.04 + CUDA 13.2 + Blackwell/RTX 50XX + Python 3.12** через TensorRT-LLM. Каноническая документация: [README](README.md), [архитектура](architecture.md), [правила](rules.md), [план](plan.md) и [журнал](changelog.md).

Рабочая GPU-машина: `192.168.0.206`, пользователь `master`, Ubuntu 26.04, RTX 5060 Ti (Blackwell, CC 12.0), 16 GiB VRAM, driver 595.71.05 и CUDA runtime/toolkit 13.2. Пароли, HF tokens, SSH host keys и иные секреты не записывай в Git, конфиги или диагностику. Перед любой GPU-работой повторно проверь фактический `nvidia-smi` и импорт PyTorch/TensorRT-LLM.

## Порядок работы

1. Прочитай `README.md`, `architecture.md`, `rules.md`, `plan.md` и актуальный `changelog.md`.
2. Проверь рабочее дерево и текущую ветку; не затирай чужие изменения.
3. До изменения логики запусти или подготовь диагностический шаг, который подтверждает причину проблемы.
4. Делай ровно один самостоятельный спринт в одной ветке `sprint/<номер>-<краткое-имя>`.
5. Добавь/обнови тесты и диагностические поля, если меняется наблюдаемое поведение.
6. Обнови `plan.md`, затем **добавь** новую запись в конец `changelog.md`; существующие записи не переписывай.
7. Выполни локальную проверку, сделай commit, push, PR и merge в `main` согласно `rules.md`.

## Неподвижные требования

- Только native Ubuntu; Docker запрещён.
- Целевой стек: Ubuntu 26.04, CUDA Toolkit 13.2, Blackwell/RTX 50XX, Python 3.12, PyTorch cu132.
- PyTorch устанавливается командой `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`.
- Не устанавливай CUDA 13.3 и не разрешай pip заменить `torch 2.13.0+cu132` ради TensorRT-LLM. При несовместимости pre-built wheel остановись до загрузки моделей и сохрани environment report.
- Каждый transformer inference использует GPU 0; `batch_size=1`. Не запускай одновременно несколько model variants на 16 GiB VRAM.
- Предпочтительный вариант — ApacheOne `full` NVFP4. `txtattn_bf16` — отдельная диагностическая проверка, а не бесшумная замена `full`.
- CPU допустим только для файлового I/O, токенизации/метаданных и orchestration; нейросетевые вычисления и тензоры не должны переходить на CPU fallback.
- Не добавляй Diffusers fallback, ComfyUI, Telegram, очереди или multi-worker.
- Не запускай генерацию как acceptance на GPU старше Blackwell.
- Не меняй TextEncoder cache, image preprocessing или runtime layout без новой подтверждённой причины.
- Не скрывай исключения: каждый существенный сбой должен сохранять диагностический JSON с traceback и версионным контекстом.

## Режимы генерации

| Режим | Назначение | Обязательное поведение |
| --- | --- | --- |
| `visualgen_prompt_text` | Только smoke-test публичного VisualGen | `prompt_cache_used=false`, `smoke_test_only=true` |
| `cached_embeddings_strict` | Целевая архитектура | Использует только `prompt_tensors.safetensors`; не кодирует текст заново и не переключается на prompt text |

Если строгий режим сообщает, что внешние embeddings не поддержаны, это ожидаемое ограничение публичного API, а не повод переделывать text encoder.

## Принцип классификации сбоев

- Не работает официальная модель VisualGen → исследуй окружение, TensorRT-LLM, CUDA, драйвер, VRAM и доступ к модели.
- Официальная модель работает, ApacheOne нет → исследуй layout/loader ApacheOne.
- Prompt-text работает → это не подтверждает strict cached embeddings архитектуру.
- Strict mode отклоняет external embeddings → сохрани отчёт и переходи к нижнеуровневому адаптеру только в отдельном спринте.

## Что включать в отчёты

Минимум: GPU и compute capability, объём/свободная VRAM, Ubuntu, драйвер, CUDA runtime, Python executable/version, PyTorch, TensorRT, TensorRT-LLM, `LD_LIBRARY_PATH`, вариант модели, режим, факт использования cache, stdout/stderr, классификаторы OOM/unsupported architecture/missing model/invalid safetensors и полный traceback.
