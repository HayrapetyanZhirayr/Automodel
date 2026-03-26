set -euo pipefail

VENV_DIR="${1:?VENV_DIR is required}"
AUTOMODEL_DIR="${2:?AUTOMODEL_DIR is required}"

# install deps
apt update && apt install -y libibverbs-dev

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

# clean-up & activate
rm -rf "$VENV_DIR"
/usr/local/bin/python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# install env
cd "$AUTOMODEL_DIR"
uv sync --extra all --extra fa --extra moe --extra cuda_source