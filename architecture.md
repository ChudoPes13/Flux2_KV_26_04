# Архитектура проекта

## Назначение

Проект исследует запуск **FLUX.2 Klein 9B-KV NVFP4** через TensorRT-LLM VisualGen или нижнеуровневый TensorRT-LLM pipeline. Главная задача — получить доказуемую совместимость и диагностируемый результат на target hardware, а не обойти ограничения fallback-ами.

## Целевая платформа

| Компонент | Целевое значение |
| --- | --- |
| ОС | Ubuntu 26.04 LTS, native runtime |
| GPU | NVIDIA Blackwell / GeForce RTX 50XX или новее |
| CUDA | 13.2 |
| Python | 3.14 |
| PyTorch | CUDA 13.2 wheels |
| Inference runtime | TensorRT + TensorRT-LLM, нативно установленный и совместимый с GPU/драйвером |
| Контейнеры | Не используются |

Базовая установка PyTorch:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

TensorRT и TensorRT-LLM не фиксируются произвольными версиями: они должны устанавливаться по актуальной официальной инструкции NVIDIA именно для CUDA 13.2, версии драйвера и Blackwell. Любой environment gate фиксирует реальные версии и пути библиотек. CUDA 13.3 не является допустимой заменой CUDA 13.2.

## Подтверждённая удалённая машина

| Поле | Фактическое значение на 2026-07-15 |
| --- | --- |
| Host | `192.168.0.206` (пользователь `master`) |
| ОС | Ubuntu 26.04 LTS, kernel 7.0.0-27-generic |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Compute capability | 12.0 (Blackwell) |
| VRAM | 16 311 MiB, свободно 15 814 MiB на момент проверки |
| Driver / CUDA runtime | 595.71.05 / 13.2 |
| RAM / свободный диск `$HOME` | 60 GiB / 811 GiB после online LVM expansion |
| Project Python | 3.14.4; `.venv` и `.venv-modelopt` используют этот interpreter |
| CUDA Toolkit | 13.2.78 (`/usr/local/cuda-13.2`, `nvcc` доступен) |
| PyTorch | 2.13.0+cu132, `torch.version.cuda == 13.2`, GPU available |
| TensorRT | 11.1.0.106, Python import проходит |
| TensorRT-LLM / VisualGen | Source-build `1.3.0rc20`; import и runtime smoke-test проходят |

GPU подходит для NVFP4 compatibility-диагностики. CUDA Toolkit, `CUDA_HOME`, TensorRT-LLM, VisualGen, NIXL и UCX проверены environment gate. Официальный BFL FLUX.2 Klein 9B-KV указывает около 29 GiB VRAM, поэтому на 16 GiB GPU его стандартный путь не является гарантированным acceptance; ApacheOne NVFP4 проверяется раздельно и без CPU fallback.

## GPU-first и точность

`full` ApacheOne NVFP4 checkpoint — первичный вариант. NVFP4 применяется к transformer ровно в той форме, в которой опубликован checkpoint; проект не делает самодельную переквантизацию до прохождения load-tests. VAE, токенизатор и другие companion-компоненты используют точность, требуемую их реализацией: нельзя принудительно объявлять их NVFP4, если runtime этого не поддерживает.

Все нейросетевые расчёты — encoding, transformer/denoising, VAE decode и tensor transforms — выполняются на GPU 0. CPU разрешён лишь для I/O, YAML/JSON, safetensors-header inspection и orchestration. На 16 GiB VRAM применяются `batch_size=1`, последовательные варианты модели и освобождение памяти между проверками; нет CPU fallback, параллельных генераций или скрытого изменения precision.

## Совместимость native TensorRT-LLM

Рабочий runtime — source-build TensorRT-LLM `v1.3.0rc20` под Python 3.14, CUDA Toolkit 13.2 и обязательный `torch 2.13.0+cu132`. Pre-built wheel не использовался, поскольку его metadata-конфликты с этим Torch не допускают resolver без downgrade. Main runtime находится только в `.venv` и активируется `scripts/activate_remote.sh`.

NVIDIA ModelOpt изолирован в `.venv-modelopt` вместе с совместимым Transformers и Hugging Face CLI; этот venv активируется `scripts/activate_modelopt_remote.sh`. Минимальная compatibility-зависимость ModelOpt остаётся в `.venv`, поскольку RC20 выполняет eager import при импорте `tensorrt_llm`. Не обновляйте и не удаляйте её без полной повторной проверки TensorRT-LLM.

Детальная процедура приведена в [workflow.md](workflow.md).

## Логическая схема

```mermaid
flowchart TD
    E[Native Ubuntu diagnostics] --> C[CPU layout/cache validation]
    C --> V[Official VisualGen support gate]
    V -->|pass| A[ApacheOne load-only checks]
    V -->|fail| R[Environment report and stop]
    A -->|both classified| P[visualgen_prompt_text smoke-test]
    P --> S[cached_embeddings_strict]
    S -->|external embeddings unsupported| L[Plan lower-level adapter]
    S -->|proceeds| O[Save verified report/output]
```

Следующий шаг не начинается до того, как предыдущий дал сохранённый и классифицированный результат.

## Слои исходного кода

```text
src/flux2_kv/
  config.py                 Загрузка и проверка configs/project.yaml
  env.py                    Native Ubuntu/CUDA/GPU/TensorRT диагностика
  diagnostics.py            JSON-отчёты, stdout/stderr и traceback
  runtime_layout.py         Сборка VisualGen-compatible layout
  runtime_validation.py     CPU-only проверка layout
  checkpoint_inspection.py  Проверка заголовков ApacheOne safetensors
  prompt_cache.py           Чтение/запись prompt_tensors.safetensors
  generation_inputs.py      Строгая валидация входов генерации
  image_io.py               Нормализация фото и logo
  pipeline_adapter.py       Граница VisualGen/low-level TensorRT-LLM
  report.py                 Run report
scripts/
  00_ubuntu_check.py
  01_download_models.py
  02_encode_prompt.py
  03_prepare_inputs.py
  validate_runtime_dir.py
  inspect_apacheone_checkpoint.py
  mock_low_level_adapter_test.py
  check_visualgen_supported_model.py
  check_visualgen_load.py
  04_generate_once.py
  rtx50_first_run_check.py
```

Имена служат контрактом целевого состояния; файлы реализуются последующими спринтами согласно [plan.md](plan.md).

## Данные и модели

```text
models/
  apacheone/
    flux2-klein-9b-kv-nvfp4.safetensors
    flux2-klein-9b-kv-nvfp4_txtattnBF16.safetensors
  bfl/                       Companion configs/tokenizer/VAE
  experimental/text_encoder/aifeifei_4bit/
data/
  input/                     prompt.txt, user_photo.png, logo.png
  cache/images/              Normalized images
  cache/prompt/main_prompt_aifeifei_4bit/
  cache/visualgen_runtime/<variant>/
  diagnostics/
  output/
```

Большие веса, private inputs, кэш и результаты не коммитятся. Для runtime layout используется symlink на ApacheOne checkpoint, а не его дубликат.

Репозитории моделей:

- Base: `black-forest-labs/FLUX.2-klein-9b-kv`
- ApacheOne NVFP4: `ApacheOne/FLUX.2-klein-9b-kv-nvfp4_mixed`
- Экспериментальный text encoder: `aifeifei798/FLUX.2-klein-9B-text_encoder-4bit`

## Prompt cache и режимы

Канонический cache: `data/cache/prompt/main_prompt_aifeifei_4bit/prompt_tensors.safetensors`.

Обязательные tensors:

```text
prompt_embeds  [1, 512, 12288]  torch.bfloat16
text_ids       [1, 512, 4]      torch.int64
```

Есть ровно два режима:

1. `visualgen_prompt_text` — временный smoke-test публичного API, который может передать prompt text. В отчёте обязательны `prompt_cache_used=false` и `smoke_test_only=true`.
2. `cached_embeddings_strict` — итоговый путь, который читает только cache и не преобразует embeddings обратно в текст. Нельзя использовать prompt-text или Diffusers fallback.

Отказ public VisualGen принять external prompt embeddings — ожидаемый и диагностически полезный результат. Он указывает на API/adapter boundary, а не на необходимость менять сформированный cache.

## Контракты диагностики

Любая значимая проверка сохраняет JSON с полями: GPU name/capability, Blackwell eligibility, VRAM total/free, Ubuntu, NVIDIA driver, CUDA runtime, Python executable/version, PyTorch, TensorRT, TensorRT-LLM, `LD_LIBRARY_PATH`, модель, вариант, режим, использование cache, stdout/stderr (или пути и tail), детекторы OOM/unsupported arch/missing model/invalid safetensors и traceback.

Итог первого target run: `data/diagnostics/rtx50_first_run_report.json` с результатами official VisualGen, ApacheOne `full`, ApacheOne `txtattn_bf16`, prompt-text smoke-test и strict cached embeddings.

## Запрещённые архитектурные обходы

- Docker и любые container-only gates;
- скрытый Diffusers fallback;
- скрытый prompt-text fallback в strict mode;
- ComfyUI, Telegram, очереди и multi-worker;
- самостоятельная quantization до завершения Blackwell load-tests;
- признание результата GPU старее Blackwell NVFP4 acceptance.
