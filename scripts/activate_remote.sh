#!/usr/bin/env bash
# Activate the native Ubuntu 26.04 environment for FLUX.2 Klein-KV via vLLM.
# Run with: source scripts/activate_remote.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Run this script with: source scripts/activate_remote.sh" >&2
  exit 1
fi

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="/usr/local/cuda-13.2"
VENV_PATH="${PROJECT_ROOT}/.venv"
NCCL_LIB_DIR="${VENV_PATH}/lib/python3.14/site-packages/nvidia/nccl/lib"
TORCH_LIB_DIR="${VENV_PATH}/lib/python3.14/site-packages/torch/lib"
VLLM_LIB_DIR="${VENV_PATH}/lib/python3.14/site-packages/vllm"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA Toolkit 13.2 was not found at ${CUDA_HOME}." >&2
  return 1
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Project venv was not found at ${VENV_PATH}." >&2
  echo "Create it with: python3.14 -m venv .venv && source .venv/bin/activate && pip install -U pip" >&2
  return 1
fi

export CUDA_HOME
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
export CUDAHOSTCXX="/usr/bin/g++-14"
export CC="/usr/bin/gcc-14"
export CXX="/usr/bin/g++-14"
# vLLM image generation on Blackwell may need an explicit attention backend;
# Sprint 101 fixes the exact value. Default FLASHINFER is the most likely candidate.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${TORCH_LIB_DIR}:${VLLM_LIB_DIR}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

source "${VENV_PATH}/bin/activate"

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_VERSION}" != "3.14" ]]; then
  echo "Expected the project venv to use Python 3.14; found ${PYTHON_VERSION}." >&2
  return 1
fi

echo "Activated $(python --version) with CUDA Toolkit $(nvcc --version | awk '/release/{print $6}') on GPU ${CUDA_VISIBLE_DEVICES}."
