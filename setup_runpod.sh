#!/bin/bash
# RunPod setup script for RF-LeWM training
# Usage: bash setup_runpod.sh
set -e

DATA_DIR="/workspace/data"
REPO_DIR="/workspace/le-rf-wm"

echo "=== 1. Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== 2. Clone repo ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/ozlabsai/le-rf-wm.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== 3. Setup Python environment ==="
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env] h5py wandb huggingface-hub[hf_xet]

echo "=== 4. Download dataset from HuggingFace ==="
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/train.h5" ]; then
    python -m huggingface_hub.commands.huggingface_cli download OzLabs/rf-spectral-trajectories \
        --repo-type dataset \
        --local-dir "$DATA_DIR"
    echo "Dataset downloaded to $DATA_DIR"
else
    echo "Dataset already exists at $DATA_DIR"
fi

ls -lh "$DATA_DIR"/*.h5

echo "=== 5. Run smoke test ==="
python smoke_test.py --data_path "$DATA_DIR/train.h5"

echo ""
echo "=== Setup complete ==="
echo "To train:"
echo "  cd $REPO_DIR && source .venv/bin/activate"
echo "  python train_rf.py data.train_path=$DATA_DIR/train.h5 data.val_path=$DATA_DIR/val.h5 wandb.enabled=False"
echo ""
echo "To train with wandb:"
echo "  wandb login"
echo "  python train_rf.py data.train_path=$DATA_DIR/train.h5 data.val_path=$DATA_DIR/val.h5"