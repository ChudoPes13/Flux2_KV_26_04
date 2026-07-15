#!/usr/bin/env bash
# Activate the native Ubuntu 26.04 environment for FLUX.2 Klein-KV.
# Run with: source scripts/activate_remote.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run this script with: source scripts/activate_remote.sh" >&2
  exit 1
fi

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="/usr/local/cuda-13.2"
VENV_PATH="${PROJECT_ROOT}/.venv"
NIXL_LIB_DIR="/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu"
UCX_LIB_DIR="/usr/local/ucx/lib"
NCCL_LIB_DIR="${VENV_PATH}/lib/python3.14/site-packages/nvidia/nccl/lib"
TORCH_LIB_DIR="${VENV_PATH}/lib/python3.14/site-packages/torch/lib"
TRTLLM_LIB_DIR="${VENV_PATH}/lib/python3.14/site-packages/tensorrt_llm/libs"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA Toolkit 13.2 was not found at ${CUDA_HOME}." >&2
  return 1
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Project venv was not found at ${VENV_PATH}." >&2
  return 1
fi

export CUDA_HOME
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
export CUDAHOSTCXX="/usr/bin/g++-14"
export CC="/usr/bin/gcc-14"
export CXX="/usr/bin/g++-14"
export LD_LIBRARY_PATH="${NIXL_LIB_DIR}:${UCX_LIB_DIR}:${NCCL_LIB_DIR}:${TORCH_LIB_DIR}:${TRTLLM_LIB_DIR}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source "${VENV_PATH}/bin/activate"

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.14" ]]; then
  echo "Expected the project venv to use Python 3.14; found ${PYTHON_VERSION}." >&2
  return 1
fi

echo "Activated $(python --version) with CUDA Toolkit $(nvcc --version | awk '/release/{print $6}') on GPU ${CUDA_VISIBLE_DEVICES}."
