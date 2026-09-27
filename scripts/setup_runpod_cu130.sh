#!/usr/bin/env bash
set -euo pipefail

# Verified on the Runpod PyTorch Ubuntu 24.04 image with a CUDA 13-capable driver.
python3 -m pip install --break-system-packages 'uv==0.12.19'
uv pip install --system --break-system-packages --torch-backend=cu130 \
  torch==2.13.0 torchvision==0.28.0 transformers==5.17.0 peft==0.21.0 \
  accelerate safetensors soundfile librosa pyyaml huggingface-hub
uv pip uninstall --system --break-system-packages torchaudio || true
python3 - <<'PY'
import torch, transformers, peft
assert torch.__version__.startswith('2.13.0'), torch.__version__
assert transformers.__version__ == '5.17.0', transformers.__version__
assert peft.__version__ == '0.21.0', peft.__version__
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), flush=True)
PY
