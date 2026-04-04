VENV_DIR="${1:?VENV_DIR is required}"
AUTOMODEL_DIR="${2:?AUTOMODEL_DIR is required}"
BASE_MODEL="${3:?BASE_MODEL is required}"
CKPT_SHARDED="${4:?CKPT_SHARDED is required}"
CKPT_CONSOLIDATED="${5:?CKPT_CONSOLIDATED is required}"
BACKEND="${6:-gloo}"
NUM_THREADS="${7:-16}"
source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/python" "$AUTOMODEL_DIR/tools/offline_hf_consolidation.py" \
    -m $BASE_MODEL \
    -i $CKPT_SHARDED \
    -o $CKPT_CONSOLIDATED \
    --backend $BACKEND \
    --num-threads $NUM_THREADS \