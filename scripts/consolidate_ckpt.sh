VENV_DIR="${1:?VENV_DIR is required}"
AUTOMODEL_DIR="${2:?AUTOMODEL_DIR is required}"
BASE_MODEL="${3:?BASE_MODEL is required}"
CKPT_SHARDED="${4:?CKPT_SHARDED is required}"
CKPT_CONSOLIDATED="${5:?CKPT_CONSOLIDATED is required}"
NPROC_PER_NODE="${6:-8}"
NTHREADS_PER_PROC="${7:-8}"

source "$VENV_DIR/bin/activate"

CUDA_VISIBLE_DEVICES="" torchrun \
  --nnodes=1 \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$(hostname) \
  --master_port=29500 \
    "$AUTOMODEL_DIR/tools/offline_hf_consolidation.py" \
    -m $BASE_MODEL \
    -i $CKPT_SHARDED \
    -o $CKPT_CONSOLIDATED \
    --backend gloo \
    --num-threads $NTHREADS_PER_PROC \