#!/bin/bash
set -e
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q "transformers>=4.51.0" accelerate sentencepiece
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "SETUP_DONE"
