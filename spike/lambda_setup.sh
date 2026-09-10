#!/bin/bash
set -e
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip
# NOTE: newer transformers requires torch>=2.6 for security reasons (torch.load
# vulnerability check) -- found this the hard way on the embedding run, install
# the right version directly this time instead of the 2-step dance.
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers adapters numpy scipy scikit-learn rank_bm25
python3 -c "import torch; print('torch cuda available:', torch.cuda.is_available())"
echo "SETUP_DONE"
