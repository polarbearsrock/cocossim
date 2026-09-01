#!/usr/bin/env bash
# Runs ON the TPU VM. Idempotent bootstrap: uv, py3.12 venv, vllm-tpu (pulls
# JAX for TPU), model prefetch. ~5 minutes on a fresh VM, seconds on reuse.
set -euo pipefail
cd "$HOME"
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
[ -d "$HOME/venv" ] || uv venv "$HOME/venv" --python 3.12
source "$HOME/venv/bin/activate"
python -c "import vllm" 2>/dev/null || uv pip install -q vllm-tpu
python - <<'EOF'
import jax
d = jax.devices()
assert d and d[0].platform == "tpu", f"no TPU visible: {d}"
print("TPU OK:", d[0].device_kind, "x", len(d))
EOF
# Prefetch the Phase-D model so measurement runs never pay download time.
export HF_HOME="$HOME/hf"
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.safetensors", "*.json", "*.txt"])
print("model cached")
EOF
echo SETUP_OK
