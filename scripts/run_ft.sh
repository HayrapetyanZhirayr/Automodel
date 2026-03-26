VENV_DIR="${1:?VENV_DIR is required}"
AUTOMODEL_DIR="${2:?AUTOMODEL_DIR is required}"
TRAIN_CONFIG="${3:?TRAIN_CONFIG is required}"
NPROC_PER_NODE="${4:-8}"
WANDB_CREDS_FILE="${5:?WANDB_CREDS_FILE is required}"


# locate wandb creds
export NETRC="$WANDB_CREDS_FILE"

source "$VENV_DIR/bin/activate"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# >>> IGNORE CHERRY PICKED WARNINGS >>>
export PYTHONWARNINGS="\
ignore:The pynvml package is deprecated:FutureWarning,\
ignore::UserWarning:pydantic._internal._generate_schema,\
ignore:Slicing a flattened dim from root mesh will be deprecated:UserWarning"
# <<< IGNORE CHERRY PICKED WARNINGS <<<



"$VENV_DIR/bin/python" -m torch.distributed.run --nproc-per-node=$NPROC_PER_NODE \
  "$AUTOMODEL_DIR/examples/llm_finetune/finetune.py" \
  -c $TRAIN_CONFIG