#!/bin/bash
set -e
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers adapters numpy scipy scikit-learn rank_bm25
python3 -c "import torch; print('torch cuda available:', torch.cuda.is_available())"
echo "SETUP_DONE"
