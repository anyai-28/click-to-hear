#!/usr/bin/env python3
"""SAM3 + SAM Audio: 動画セグメンテーション & 音声分離アプリ

Usage:
    uv run python main.py [--port PORT] [--share]

初回セットアップ:
    1. Hugging Face認証: huggingface-cli login
    2. モデルアクセス申請:
       - https://huggingface.co/facebook/sam3
       - https://huggingface.co/facebook/sam-audio-small
    3. 依存関係インストール: uv sync
"""

import os
import argparse
from pathlib import Path


def load_envrc():
    """プロジェクトルートの.envrcから環境変数を読み込む"""
    envrc_path = Path(__file__).parent / ".envrc"
    if envrc_path.exists():
        with open(envrc_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    # export VAR="value" または VAR="value" の形式を処理
                    if line.startswith("export "):
                        line = line[7:]
                    key, _, value = line.partition("=")
                    # クォートを除去
                    value = value.strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value


load_envrc()

# MPS fallback設定（Mac用）
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.config.settings import AppSettings
from src.ui.app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 + SAM Audio: 動画セグメンテーション & 音声分離"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="サーバーポート (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Gradio共有リンクを生成"
    )

    args = parser.parse_args()

    settings = AppSettings(
        server_port=args.port,
        share=args.share,
    )

    print("=" * 60)
    print("SAM3 + SAM Audio: 動画セグメンテーション & 音声分離")
    print("=" * 60)
    print()
    print("モデル設定:")
    print(f"  - SAM3: {settings.sam3_model}")
    print(f"  - BLIP-2: {settings.blip2_model}")
    print(f"  - SAM Audio: {settings.sam_audio_model}")
    print()
    print("使用方法:")
    print("  1. 動画をアップロード")
    print("  2. フレームを選択")
    print("  3. 対象物をクリック")
    print("  4. 説明文を確認・編集")
    print("  5. 「音声分離実行」をクリック")
    print()

    app = create_app(settings)
    app.launch()


if __name__ == "__main__":
    main()
