# Click to Hear

動画から対象物をクリックでセグメンテーションし、その対象物の音声を分離するアプリケーション。

> クリックで見て、聞きたい音を分離する - Visual segmentation meets audio separation

## 機能

- 動画のアップロードとフレーム選択
- クリックによる対象物のセグメンテーション（SAM3）
- セグメントされた対象物の説明文自動生成（BLIP-2）
- 説明文に基づく音声分離（SAM Audio）
- 対象音声/背景音声の再生・ダウンロード

## 使用モデル

| モデル | 用途 |
|--------|------|
| [SAM 3](https://huggingface.co/facebook/sam3) | セグメンテーション |
| [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) | 説明文生成 |
| [SAM Audio](https://huggingface.co/facebook/sam-audio-small) | 音声分離 |

ライセンス詳細は[ライセンス](#ライセンス)セクションを参照。

## 動作環境

- macOS (Apple Silicon対応)
- Python 3.12+
- uv (パッケージマネージャー)

## セットアップ

### 1. 前提条件（macOS）

Homebrewで必要なツールをインストール:

```bash
# Homebrewがない場合はインストール
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 必要なツールをインストール
brew install ffmpeg cmake llvm libomp direnv
```

direnvのシェル統合を有効化（~/.zshrcに追加）:

```bash
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
# direnvを有効化
direnv allow
source ~/.zshrc
```

### 2. Hugging Face アカウント準備

Hugging Faceのアカウントが必要です。持っていない場合は以下で作成:

- <https://huggingface.co/join>

### 3. アクセストークンの取得

1. <https://huggingface.co/settings/tokens> にアクセス
2. 「New token」をクリック
3. 名前を入力し、「Read」権限でトークンを作成
4. トークンをコピー
5. `.envrc`ファイルの`export HF_TOKEN=""`にトークンを貼り付け

### 4. モデルへのアクセス申請

以下のページでそれぞれ「Agree and access repository」をクリック:

- <https://huggingface.co/facebook/sam3>
- <https://huggingface.co/facebook/sam-audio-small>

（承認には数分〜数時間かかる場合があります）

### 5. 環境セットアップ

```bash
# リポジトリに移動
cd /path/to/click-to-hear

# Hugging Face認証（トークンを入力）
uv run hf auth login

# 依存関係インストール（時間がかかります）
uv sync
```

## 使い方

### アプリ起動

```bash
# 基本起動（ポート7860）
uv run python main.py

# ポート指定
uv run python main.py --port 8080

# 共有リンク生成（外部からアクセス可能）
uv run python main.py --share
```

起動後、ブラウザで <http://localhost:7860> にアクセス。

### 操作手順

1. **動画をアップロード** - 音声付きの動画ファイルをドラッグ&ドロップ
2. **フレームを選択** - スライダーで対象のフレームを選択
3. **対象物をクリック** - 分離したい音源の対象物をクリック
4. **説明文を確認** - 自動生成された説明文を確認・編集
5. **音声分離実行** - ボタンをクリックして音声分離を実行
6. **結果を確認** - 対象音声と背景音声をそれぞれ再生・ダウンロード

## プロジェクト構成

```text
click-to-hear/
├── main.py                      # エントリーポイント
├── pyproject.toml               # 依存関係
├── src/
│   ├── config/
│   │   ├── settings.py          # アプリ設定
│   │   └── device.py            # デバイス検出 (MPS/CPU)
│   ├── video/
│   │   └── processor.py         # 動画処理・音声抽出
│   ├── segmentation/
│   │   └── sam3_wrapper.py      # SAM3ラッパー
│   ├── captioning/
│   │   └── blip2_captioner.py   # BLIP-2ラッパー
│   ├── audio/
│   │   └── sam_audio_wrapper.py # SAM Audioラッパー
│   └── ui/
│       └── app.py               # Gradio UI
├── outputs/                     # 出力ファイル保存
└── models/                      # モデルキャッシュ
```

## トラブルシューティング

### モデルのダウンロードに失敗する

```text
Access denied for model facebook/sam3
```

→ Hugging Faceでモデルへのアクセス申請が完了しているか確認してください。

### メモリ不足エラー

→ 大きな動画は処理に多くのメモリを必要とします。短い動画で試すか、`src/config/settings.py`で`max_frames`を減らしてください。

### MPS関連のエラー（Mac）

→ `.envrc` で `PYTORCH_ENABLE_MPS_FALLBACK=1` が自動設定されます。direnvが有効か確認:

```bash
direnv allow
uv run python main.py
```

### xformersのビルドエラー

→ OpenMP関連のエラーが出る場合は、direnvが有効になっているか確認してから`uv sync`を再実行:

```bash
direnv allow
uv sync
```

## ライセンス

### このプロジェクト

MIT License

### 使用モデルのライセンス

| モデル | ライセンス | 商用利用 | 備考 |
|--------|-----------|---------|------|
| [SAM 3](https://github.com/facebookresearch/sam3) | META SAM License | ✅ 可（条件付き） | 下記参照 |
| [SAM Audio](https://github.com/facebookresearch/sam-audio) | META SAM License | ✅ 可（条件付き） | 下記参照 |
| [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) | MIT License | ✅ 可 | |

### META SAM License について

SAM 3 および SAM Audio は Meta の SAM License の下で提供されています。商用利用は許可されていますが、以下の制限があります:

**禁止事項:**

- 軍事・戦争目的での使用
- 核関連・スパイ活動・武器開発への使用
- リバースエンジニアリング
- 輸出規制（ITAR）対象国・組織への提供

**義務:**

- 再配布時はライセンス文書の添付が必要
- 研究論文では SAM Materials の使用を明記
- プライバシー・データ保護法への準拠

詳細: <https://github.com/facebookresearch/sam-audio/blob/main/LICENSE>

### 依存ライブラリのライセンス

| ライブラリ | ライセンス | 商用利用 |
|-----------|----------|---------|
| PyTorch | BSD-3-Clause | ✅ 可 |
| Transformers | Apache 2.0 | ✅ 可 |
| Gradio | Apache 2.0 | ✅ 可 |
| OpenCV | Apache 2.0 / BSD | ✅ 可 |
| MoviePy | MIT | ✅ 可 |
| NumPy | BSD | ✅ 可 |
| Pillow | HPND | ✅ 可 |
| Hugging Face Hub | Apache 2.0 | ✅ 可 |
| Accelerate | Apache 2.0 | ✅ 可 |
