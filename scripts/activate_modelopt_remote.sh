#!/usr/bin/env bash
# Source this file: source scripts/activate_modelopt_remote.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this script instead: source scripts/activate_modelopt_remote.sh" >&2
  exit 1
fi

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="/usr/local/cuda-13.2"
VENV_PATH="$PROJECT_ROOT/.venv-modelopt"
TORCH_LIB_DIR="$VENV_PATH/lib/python3.14/site-packages/torch/lib"

[[ -x "$VENV_PATH/bin/python" ]] || { echo "ModelOpt venv not found: $VENV_PATH" >&2; return 1; }
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "CUDA 13.2 not found: $CUDA_HOME" >&2; return 1; }

export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
export CUDAHOSTCXX=/usr/bin/g++-14
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

source "$VENV_PATH/bin/activate"

python -c 'import sys; assert sys.version_info[:2] == (3, 14), sys.version'
echo "Activated isolated NVIDIA ModelOpt environment: Python $(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))') with CUDA Toolkit $(nvcc --version | awk '/release/ {print $6}' | tr -d ',')."
