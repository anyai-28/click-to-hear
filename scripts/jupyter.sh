#!/bin/bash
# Jupyter起動スクリプト（FFmpegライブラリパス設定済み）

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg/lib:$DYLD_LIBRARY_PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$(dirname "$0")/.."
uv run jupyter lab
