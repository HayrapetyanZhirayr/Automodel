set -euo pipefail

VENV_DIR="${1:?VENV_DIR is required}"
AUTOMODEL_DIR="${2:?AUTOMODEL_DIR is required}"

DEEP_EP_REV="e3908bf5bd0cc6265bcb225d15cd8c996d4759ef"
DEEP_EP_URL="git+https://github.com/deepseek-ai/DeepEP.git@${DEEP_EP_REV}"

# install deps
apt update && apt install -y libibverbs-dev

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

# clean-up & activate
rm -rf "$VENV_DIR"
/usr/local/bin/python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Detect GPU compute capability to decide the deep_ep build strategy.
# deep_ep uses compile-time code paths: SM90 features (FP8, TMA, aggressive
# PTX) are unavailable on SM80 (A100), so we must disable them for that arch.
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "Detected GPU compute capability: ${GPU_CC}"

cd "$AUTOMODEL_DIR"
uv cache clean deep-ep

if [ "${GPU_CC}" -lt "90" ]; then
    echo "GPU is pre-Hopper (sm_${GPU_CC}): will build deep_ep separately for A100"

    # Step 1: install everything except deep_ep (skip --extra moe)
    uv sync --extra all --extra fa --extra cuda_source

    # Step 2: build deep_ep for SM80.
    # deep_ep setup.py asserts that nvshmem must NOT be importable when
    # DISABLE_SM90_FEATURES=1 (internode kernels require SM90).
    # nvidia-nvshmem-cu12 is already in the venv (pulled by torch/TE),
    # so we temporarily hide it during the build.
    _nvshmem=$("$VENV_DIR/bin/python" -c \
        "import nvidia.nvshmem; print(nvidia.nvshmem.__path__[0])" 2>/dev/null || true)
    if [ -n "$_nvshmem" ]; then
        mv "$_nvshmem" "${_nvshmem}._hidden"
        export LD_LIBRARY_PATH="${_nvshmem}._hidden/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    DISABLE_SM90_FEATURES=1 TORCH_CUDA_ARCH_LIST="${GPU_CC:0:1}.${GPU_CC:1}" \
        pip install --no-deps --no-build-isolation --force-reinstall \
        "deep_ep @ ${DEEP_EP_URL}"

    [ -n "$_nvshmem" ] && mv "${_nvshmem}._hidden" "$_nvshmem"
else
    echo "GPU is Hopper+ (sm_${GPU_CC}): building deep_ep with SM90 features enabled"

    # Install everything except moe first so that nvidia-nvtx wheels are
    # already in the venv before we attempt to build deep-ep from source.
    uv sync --extra all --extra fa --extra cuda_source

    # deep-ep's hybrid_ep extension links against -lnvtx3interop at build time.
    # The nvidia-nvtx Python package ships only libnvtx3interop.so.1 (no
    # unversioned symlink), and its lib dir is not on the system linker path.
    # Fix both issues before the moe sync triggers the deep-ep build.
    _nvtx_lib=$("$VENV_DIR/bin/python" -c \
        "import os, nvidia.nvtx; print(os.path.join(nvidia.nvtx.__path__[0], 'lib'))" 2>/dev/null || true)
    if [ -n "$_nvtx_lib" ] && [ -f "${_nvtx_lib}/libnvtx3interop.so.1" ]; then
        [ -e "${_nvtx_lib}/libnvtx3interop.so" ] || \
            ln -s "${_nvtx_lib}/libnvtx3interop.so.1" "${_nvtx_lib}/libnvtx3interop.so"
        export LIBRARY_PATH="${_nvtx_lib}${LIBRARY_PATH:+:$LIBRARY_PATH}"
        echo "Registered nvtx lib dir for linker: ${_nvtx_lib}"
    else
        echo "Warning: libnvtx3interop.so.1 not found; deep-ep build may fail"
    fi

    uv sync --extra all --extra fa --extra moe --extra cuda_source
fi