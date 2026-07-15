# Успешная нативная установка на Ubuntu 26.04

Этот документ фиксирует фактическую конфигурацию, на которой 15 июля 2026 года успешно импортировались TensorRT-LLM 1.3.0rc20, NIXL и VisualGen. Docker не используется.

## Итоговая матрица версий

| Компонент | Установленная версия |
| --- | --- |
| ОС | Ubuntu 26.04 |
| Python | 3.14.4 |
| GPU | NVIDIA GeForce RTX 5060 Ti, SM 12.0 |
| CUDA Toolkit / nvcc | 13.2.78 |
| PyTorch | `2.13.0+cu132` |
| torchvision | `0.28.0+cu132` |
| TensorRT C++ SDK | `11.0.0.114-1+cuda13.2` |
| TensorRT Python | `11.1.0.106` |
| TensorRT-LLM | `v1.3.0rc20`, commit `c25c23f71786bad54d192893d696ce8043426eca` |
| UCX | 1.21.0, commit `167a4c6a311d9a42e30a37dcc01b8a3e73ea2826` |
| NIXL | v1.0.1, commit `d196ff6eee7217b5958cc7efcbbd5928599e252a` |

## 1. Базовые системные зависимости

На Ubuntu установлены нативные инструменты сборки, Python 3.14 и RDMA-зависимости:

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.14 python3.14-dev python3.14-venv \
  build-essential gcc-14 g++-14 cmake ninja-build ccache git-lfs pkg-config \
  libopenmpi-dev libibverbs-dev rdma-core librdmacm-dev libnuma-dev libzmq3-dev \
  patchelf
```

CUDA Toolkit должен быть именно в `/usr/local/cuda-13.2`. Не устанавливайте CUDA 13.3 или её APT-пакеты.

## 2. TensorRT и CUDA 13.2

С [страницы TensorRT 11.x](https://developer.nvidia.com/tensorrt/download/11x) был установлен локальный Debian-репозиторий NVIDIA для TensorRT 11.0 и CUDA 13.2. После импорта его ключа установлены runtime, dev и Python-пакеты TensorRT. Проверка:

```bash
nvcc --version
dpkg-query -W -f='${Version}\n' libnvinfer11
```

Ожидаются `release 13.2` и `11.0.0.114-1+cuda13.2`. Python binding `tensorrt==11.1.0.106` установлен отдельно для Python 3.14; C++-сборка связывается с системным SDK CUDA 13.2.

## 3. Чистое окружение и обязательный PyTorch

Старые проектные venv удаляются, затем создаётся единственный `.venv`:

```bash
cd ~/Flux2_KV_26_04
find . -maxdepth 1 -type d -name '.venv*' -prune -exec rm -rf {} +
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Не позволяйте resolver TensorRT-LLM понизить эти пакеты. Успешная проверка:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

## 4. UCX и NIXL

UCX собран из `v1.21.x` с CUDA 13.2, verbs и RDMA и установлен в `/usr/local/ucx`. Затем NIXL `v1.0.1` собран с UCX plugin и установлен в `/opt/nvidia/nvda_nixl`.

```bash
echo /usr/local/ucx/lib | sudo tee /etc/ld.so.conf.d/ucx.conf
echo /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu | sudo tee /etc/ld.so.conf.d/nixl.conf
sudo ldconfig
```

Использованы именно теги и commit из таблицы выше. При повторной сборке берите параметры UCX и NIXL из `docker/common/install_ucx.sh` и `docker/common/install_nixl.sh` тега TensorRT-LLM `v1.3.0rc20`, но запускайте их нативно, без Docker.

## 5. Сборка TensorRT-LLM из исходников

```bash
git clone --branch v1.3.0rc20 --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git ~/src/TensorRT-LLM-1.3.0rc20
cd ~/src/TensorRT-LLM-1.3.0rc20
```

Для данной комбинации версий исходники RC20 были минимально адаптированы:

1. `StreamReader` переведён на TensorRT 11 `IStreamReaderV2`; execution context — на `kUSER_MANAGED`; profile values — на V2 API.
2. Убраны конфликтующие forward declaration `ILogger`/`ILoggerFinder` для TensorRT 11.
3. Устаревшая проверка `allInputShapesSpecified` ограничена TensorRT 10.
4. CUTLASS `setup_library.py develop --user` заменён на `develop`, чтобы работать из venv.
5. В заголовке Torch исправлено сравнение `ArrayRef` с `OptionalArrayRef`: `self->strides() == *stride`.
6. `requirements-dev.txt` не должен устанавливать RC20 constraints, которые принудительно меняют Torch и TensorRT.

Сборка выполнялась GCC 14, SM120, системным TensorRT и NCCL из PyTorch:

```bash
source ~/Flux2_KV_26_04/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
export CUDAHOSTCXX=/usr/bin/g++-14
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export CMAKE_PREFIX_PATH=/usr/local/ucx
export PKG_CONFIG_PATH=/usr/local/ucx/lib/pkgconfig
export LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/usr/local/ucx/lib
export NCCL_ROOT=~/Flux2_KV_26_04/.venv/lib/python3.14/site-packages/nvidia/nccl
export CPATH="$NCCL_ROOT/include"

python scripts/build_wheel.py --no-venv --generator Ninja \
  --cuda_architectures=120-real --job_count 4 \
  --trt_root /usr --nccl_root "$NCCL_ROOT" --nixl_root /opt/nvidia/nvda_nixl \
  --install \
  --extra-cmake-vars NCCL_LIBRARY="$NCCL_ROOT/lib/libnccl.so.2" \
  --extra-cmake-vars NCCL_INCLUDE_DIR="$NCCL_ROOT/include" \
  --extra-cmake-vars CMAKE_PREFIX_PATH=/usr/local/ucx \
  --extra-cmake-vars ENABLE_UCX=ON
```

Если builder останавливается на вызове `python -m build` из-за одноимённой папки `build/` в корне исходников, запускайте wheel builder из нейтральной директории:

```bash
cd /tmp
~/Flux2_KV_26_04/.venv/bin/python -m pip install build
~/Flux2_KV_26_04/.venv/bin/python -m build ~/src/TensorRT-LLM-1.3.0rc20 \
  --skip-dependency-check --no-isolation --wheel \
  --outdir ~/src/TensorRT-LLM-1.3.0rc20/build
find ~/src/TensorRT-LLM-1.3.0rc20/build -name 'tensorrt_llm-*.whl' -print0 \
  | xargs -0 -r ~/Flux2_KV_26_04/.venv/bin/python -m pip install --force-reinstall --no-deps
```

## 6. Runtime Python-зависимости

Поскольку RC20 metadata требует несовместимые Torch/TensorRT, зависимости ставятся отдельно, без переустановки фиксированных GPU-пакетов. Для текущего успешного импорта потребовались как минимум `transformers`, `diffusers`, `accelerate`, `nvidia-modelopt`, `flashinfer-python`, `nvidia-cutlass-dsl[cu13]`, `apache-tvm-ffi`, `llguidance`, `xgrammar`, `cache-dit`, `mistral-common`, `onnx-graphsurgeon`, `grpcio`, `smg-grpc-proto` и их Python-зависимости.

`cuda-python` и `cuda-bindings` закреплены на `13.2.0`; проверяйте это после любого `pip install`:

```bash
python -m pip list --format=freeze | grep -E '^(cuda-python|cuda-bindings|cuda-toolkit)='
```

## Ограничения

- Это рабочая compatibility-сборка, а не официально поддерживаемая NVIDIA комбинация RC20 + TensorRT 11 + Torch 2.13.
- `pip check` будет сообщать об ожидаемых конфликтах metadata RC20 с явно заданными Torch, TensorRT и свежими runtime-пакетами. Не исправляйте их автоматическим downgrade.
- Smoke-тест не строит TensorRT engine и не запускает модель: для этого нужны выбранные веса и отдельный acceptance-тест.
- NIXL ускоряет передачу KV-cache в distributed/RDMA-сценариях. На одной GPU он не ускоряет собственно kernels inference.
