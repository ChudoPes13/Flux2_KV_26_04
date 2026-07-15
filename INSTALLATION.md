# Установка vLLM runtime на Ubuntu 26.04 + RTX 5090

Этот документ описывает целевую конфигурацию, необходимую для запуска FLUX.2 Klein-KV через vLLM image generation на RTX 5090 (Blackwell, SM 12.0, 32 GiB VRAM) с Python 3.14 и CUDA 13.2. Docker не используется.

> Статус: рецепт ниже — целевой контракт. Точные commit vLLM и патчи фиксируются в Sprint 101 и добавляются отдельной записью в этот файл.

## Итоговая матрица версий (target)

| Компонент | Целевая версия |
| --- | --- |
| ОС | Ubuntu 26.04 LTS |
| Python | 3.14.4 |
| GPU | NVIDIA GeForce RTX 5090, SM 12.0, 32 GiB VRAM |
| NVIDIA driver | последний проприетарный driver, поддерживающий RTX 5090 и CUDA 13.2 |
| CUDA Toolkit / nvcc | 13.2.78 (`/usr/local/cuda-13.2`) |
| PyTorch | `2.13.0+cu132` |
| torchvision | `0.28.0+cu132` |
| vLLM | source-build (точный commit фиксируется в Sprint 101) |
| Compressed-Tensors | та же версия в `.venv` и `.venv-modelopt` (фиксируется в Sprint 101) |
| NVIDIA ModelOpt | `0.45.0` (только `.venv-modelopt`) |
| transformers | `5.9.0` (только `.venv-modelopt`) |

## 1. Базовые системные зависимости

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.14 python3.14-dev python3.14-venv \
  build-essential gcc-14 g++-14 cmake ninja-build ccache git-lfs pkg-config \
  patchelf
```

CUDA Toolkit должен быть именно в `/usr/local/cuda-13.2`. Не устанавливайте CUDA 13.3 или её APT-пакеты.

## 2. NVIDIA driver

Установите проприетарный driver, поддерживающий RTX 5090 (Blackwell) и CUDA 13.2 runtime. Проверка:

```bash
nvidia-smi
```

Ожидается: GPU `NVIDIA GeForce RTX 5090`, driver version совместим с CUDA 13.2, compute capability `12.0`.

## 3. CUDA Toolkit 13.2

Установите CUDA Toolkit 13.2.78 в `/usr/local/cuda-13.2`. Проверка:

```bash
nvcc --version
ls -l /usr/local/cuda-13.2/bin/nvcc
```

`CUDA_HOME` в `scripts/activate_remote.sh` жёстко указывает на `/usr/local/cuda-13.2`. Не подменяйте его на 13.3.

## 4. Чистое окружение и обязательный PyTorch

Старые проектные venv удаляются, затем создаётся единственный `.venv` для vLLM runtime:

```bash
cd ~/Flux2_KV_VLLM
find . -maxdepth 1 -type d -name '.venv*' -prune -exec rm -rf {} +
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Не позволяйте resolver vLLM понизить эти пакеты. Успешная проверка:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Ожидается: `2.13.0+cu132 13.2 NVIDIA GeForce RTX 5090 (12, 0)`.

## 5. Сборка vLLM из исходников

vLLM собирается из source-ветки, поддерживающей Python 3.14 + CUDA 13.2 + Blackwell SM 12.0. Точная ветка и commit фиксируются в Sprint 101.

```bash
cd ~/src
git clone https://github.com/vllm-project/vllm.git
cd vllm
# Checkout ветки/тега, зафиксированного в Sprint 101 (TODO: pin commit)
```

Сборка:

```bash
source ~/Flux2_KV_VLLM/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
export CUDAHOSTCXX=/usr/bin/g++-14
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export VLLM_TARGET_DEVICE=cuda
export MAX_JOBS=4

pip install -v -e .
```

Альтернатива через wheel (если доступен):

```bash
pip install -v .
```

Проверка:

```bash
python -c "import vllm; print(vllm.__version__)"
python - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
flux = [a for a in archs if 'flux' in a.lower()]
assert flux, 'FLUX arch not registered in vLLM'
print('flux archs:', flux)
PY
```

Если FLUX не зарегистрирован, проверьте, что в source-ветке vLLM присутствует `vllm/model_executor/models/flux.py` и его класс зарегистрирован в `vllm/model_executor/models/registry.py`. При отсутствии — выберите более свежую ветку vLLM или поднимите issue upstream.

## 6. Runtime Python-зависимости

Дополнительно к vLLM в `.venv` устанавливаются:

```bash
pip install \
  'compressed-tensors>=0.9' \
  'huggingface_hub>=0.30' \
  'Pillow>=11.0' \
  'PyYAML>=6.0' \
  'safetensors>=0.5' \
  'numpy>=2.0' \
  'scikit-image>=0.24'
```

`cuda-python` и `cuda-bindings` должны оставаться на `13.2.0`. Проверяйте это после любого `pip install`:

```bash
python -m pip list --format=freeze | grep -E '^(cuda-python|cuda-bindings|cuda-toolkit)='
```

## 7. Изолированное окружение NVIDIA ModelOpt

NVIDIA ModelOpt установлен отдельно в `~/Flux2_KV_VLLM/.venv-modelopt`. В нём сохранён обязательный PyTorch cu132 и выбран совместимый с ModelOpt набор Hugging Face пакетов: `nvidia-modelopt==0.45.0` и `transformers==5.9.0`.

```bash
cd ~/Flux2_KV_VLLM
python3.14 -m venv .venv-modelopt
source .venv-modelopt/bin/activate
python -m pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -m pip install 'nvidia-modelopt[hf]' 'transformers==5.9.0' 'compressed-tensors>=0.9'
```

Активируйте его только через `source scripts/activate_modelopt_remote.sh`. Основной `.venv` не содержит ModelOpt — это устраняет историческую проблему eager-import ModelOpt при загрузке runtime.

## 8. Compressed-Tensors совместимость

Compressed-Tensors версия должна совпадать в обоих venv, иначе NVFP4 checkpoint, созданный в `.venv-modelopt`, может не загрузиться в vLLM runtime. После установки проверьте:

```bash
for venv in .venv .venv-modelopt; do
  echo "=== $venv ==="
  ~/Flux2_KV_VLLM/$venv/bin/python -m pip show compressed-tensors | grep -E '^(Name|Version)'
done
```

Если версии расходятся — приведите их к одному значению через `pip install 'compressed-tensors==X.Y.Z'` в обоих venv.

## Ограничения

- Это целевой контракт. Реальная успешная установка фиксируется в Sprint 101 с указанием точного commit vLLM и применённых патчей.
- vLLM image generation на Blackwell SM 12.0 может требовать `VLLM_ATTENTION_BACKEND=FLASHINFER` или аналогичного. Конкретное значение фиксируется в Sprint 101.
- `pip check` может сообщать об ожидаемых конфликтах metadata — не исправляйте их автоматическим downgrade.
- Smoke-тест не строит engine и не запускает модель: для этого нужны выбранные веса и отдельный acceptance-тест.
- vLLM image generation не требует UCX/NIXL — эти библиотеки не устанавливаются в новом проекте.
