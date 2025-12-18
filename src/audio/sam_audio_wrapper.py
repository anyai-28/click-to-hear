import torch
import torchaudio
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from ..config.device import get_device_for_model


def set_seed(seed: int) -> None:
    """乱数シードを固定して再現性を確保

    Args:
        seed: 乱数シード値
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class SeparationResult:
    """音声分離結果"""

    target: torch.Tensor  # 対象音声 (samples,)
    residual: torch.Tensor  # 残り音声 (samples,)
    sample_rate: int


class SAMAudioWrapper:
    """SAM Audio 音声分離ラッパー"""

    def __init__(self, model_name: str = "facebook/sam-audio-base", seed: int = 42):
        self.model_name = model_name
        self.device = get_device_for_model("sam_audio")
        self.seed = seed

        self._model = None
        self._processor = None

    @property
    def model(self):
        """モデルを遅延ロード"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        """プロセッサを遅延ロード"""
        if self._processor is None:
            self._load_model()
        return self._processor

    def _load_model(self):
        """モデルをロード"""
        import os
        from sam_audio import SAMAudio, SAMAudioProcessor

        token = os.environ.get("HF_TOKEN")
        self._model = SAMAudio.from_pretrained(self.model_name, token=token)
        self._processor = SAMAudioProcessor.from_pretrained(self.model_name)

        self._model = self._model.to(self.device).eval()

    def separate_by_text(self, audio_path: str, description: str) -> SeparationResult:
        """テキストプロンプトで音声分離

        Args:
            audio_path: 入力音声ファイルパス
            description: 分離対象の説明文 (例: "A man speaking", "Dog barking")

        Returns:
            SeparationResult: 分離された音声（target: 対象, residual: 残り）
        """
        if self.seed is not None:
            set_seed(self.seed)

        batch = self.processor(audios=[audio_path], descriptions=[description]).to(self.device)

        with torch.inference_mode():
            result = self.model.separate(batch)

        return SeparationResult(
            target=result.target[0].cpu(),
            residual=result.residual[0].cpu(),
            sample_rate=self.processor.audio_sampling_rate,
        )

    def separate_multiple(self, audio_path: str, descriptions: list[str]) -> list[SeparationResult]:
        """複数の説明文で音声分離

        Args:
            audio_path: 入力音声ファイルパス
            descriptions: 分離対象の説明文リスト

        Returns:
            各説明文に対応する分離結果のリスト
        """
        results = []
        for desc in descriptions:
            result = self.separate_by_text(audio_path, desc)
            results.append(result)
        return results

    def unload(self):
        """モデルをアンロードしてメモリを解放"""
        self._model = None
        self._processor = None
        from ..config.device import clear_memory

        clear_memory(self.device)


def save_audio(audio_tensor: torch.Tensor, output_path: str, sample_rate: int) -> str:
    """音声をWAVファイルとして保存

    Args:
        audio_tensor: 音声テンソル (samples,) または (1, samples)
        output_path: 出力ファイルパス
        sample_rate: サンプルレート

    Returns:
        保存したファイルパス
    """
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    torchaudio.save(output_path, audio_tensor, sample_rate)
    return output_path


def load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    """音声ファイルをロード

    Args:
        audio_path: 音声ファイルパス

    Returns:
        (音声テンソル, サンプルレート)
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate
