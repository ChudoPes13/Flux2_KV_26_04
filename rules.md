# Правила разработки

## Обязательный GitHub-процесс

Репозиторий: <https://github.com/la6su/Flux2_KV_VLLM>

Каждый самостоятельный этап (спринт) выполняется строго в отдельной ветке:

```text
main
  └── sprint/<номер>-<краткое-описание>
```

Нумерация спринтов после rewrite под vLLM стартует со 100 (см. `plan.md`), чтобы избежать коллизий с историческими спринтами TensorRT-LLM era.

Обязательная последовательность:

1. Синхронизировать локальный `main` с `origin/main`.
2. Создать ветку `sprint/<номер>-<краткое-описание>` от актуального `main`.
3. Сделать только изменения текущего спринта.
4. Запустить относящиеся к изменению проверки и зафиксировать их результат в PR.
5. Обновить `plan.md` и добавить запись в конец `changelog.md`.
6. Commit с атомарным сообщением, например `feat(env): add vLLM smoke test`.
7. `git push -u origin <ветка>`.
8. Создать PR в `main`, заполнить шаблон и дождаться требуемых проверок/review.
9. Выполнить merge PR в `main`; после merge синхронизировать локальный `main`.

Прямые содержательные push в `main` запрещены. Исключение — только техническая инициализация пустого репозитория, необходимая для создания первой PR-ветки.

## Docker запрещён

Проект работает только в native Ubuntu environment. Не добавляйте Dockerfile, docker-compose, container-specific scripts, `running_in_container` gates, container fallback или документацию, предлагающую Docker как путь исполнения.

## Backend и runtime

- Единственный inference backend — vLLM (`model_type=image`, FLUX.2 dispatch).
- TensorRT-LLM, VisualGen, ONNX Runtime, MIGraphX, Diffusers pipeline-as-runtime и любые C++ engine, отличные от vLLM, запрещены.
- Diffusers разрешён только как **библиотека моделей** внутри vLLM (vLLM сам импортирует diffusers model classes), но прямой вызов `diffusers.FluxPipeline` как fallback runtime запрещён.

## Удалённая GPU-машина

- GPU-работа выполняется на целевой Ubuntu-машине с RTX 5090 (Blackwell, CC 12.0, 32 GiB VRAM); локальная Windows-машина служит только клиентом управления и Git-копией.
- Не записывайте в Git или PR SSH-пароли, токены Hugging Face, host keys, команды с секретами и приватные данные пользователя.
- Перед GPU-спринтом документируйте в diagnostic JSON фактические GPU, driver, CUDA и доступную VRAM. Не считайте прошлый замер вечным.
- Не меняйте системные драйверы, CUDA Toolkit или пакеты ОС в том же спринте, где меняется model loader или generation logic.

## Требования к качеству

- Никаких silent fallbacks и swallowed exceptions.
- До изменения гипотезы зафиксируйте диагностические данные, подтверждающие её.
- Секреты, токены Hugging Face, веса моделей, input-изображения, cache, output и diagnostics не попадают в Git.
- Изменения должны быть минимальными, проверяемыми и обратимыми в рамках одной PR.
- Добавляйте тест или диагностическую проверку к новой логике.
- Не переписывайте историю `changelog.md`; добавляйте новую запись в конец файла.

## Запреты предметной области

- Не добавлять Diffusers-fallback runtime, ComfyUI, Telegram, очереди или multi-worker.
- Не использовать ApacheOne NVFP4 pre-quantized checkpoint или сторонние 4-bit text encoder. NVFP4 checkpoint создаётся самим проектом через ModelOpt.
- Не переквантизировать модель «всё целиком» без явно задокументированного `target_modules`.
- Не менять `target_modules` / `ignore_modules` без отдельного спринта с метрическим доказательством.
- Не считать non-Blackwell GPU финальной NVFP4-проверкой.
- Не запускать NVFP4 acceptance до того, как BF16 baseline сохранён.
- Не использовать тот же venv для vLLM runtime и ModelOpt quantization.

## Соглашения об именовании скриптов

- `00_ubuntu_check.py` — environment gate.
- `01_vllm_smoke.py` — vLLM import + FLUX.2 registry probe.
- `02_modelopt_smoke.py` — ModelOpt venv sanity check.
- `03_model_readiness.py` — HF access + disk + GPU preflight.
- `04_download_models.py` — snapshot download.
- `05_bf16_baseline.py` — BF16 baseline run.
- `06_quantize_nvfp4.py` — NVFP4 quantization (ModelOpt venv).
- `07_quantized_run.py` — NVFP4 run (vLLM runtime venv).
- `08_compare_results.py` — BF16 vs NVFP4 comparison.

Скрипты нельзя «пропускать»: если следующий шаг зависит от артефакта предыдущего, предыдущий должен быть `pass`.
