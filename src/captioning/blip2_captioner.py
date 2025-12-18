import torch
import numpy as np
from PIL import Image
from typing import Optional

from ..config.device import get_device_for_model, get_dtype_for_device


class BLIP2Captioner:
    """BLIP-2を使用した画像説明文生成"""

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self.model_name = model_name
        self.device = get_device_for_model("blip2")
        self.dtype = get_dtype_for_device(self.device)

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
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        self._processor = Blip2Processor.from_pretrained(self.model_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self._model.eval()

    def generate_caption(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        prompt: Optional[str] = None,
        max_new_tokens: int = 50,
    ) -> str:
        """画像から説明文を生成

        Args:
            image: RGB形式の画像 (H, W, 3)
            mask: オプションのセグメンテーションマスク (H, W)
                  指定すると、マスク領域を切り出して処理
            prompt: オプションの質問プロンプト
            max_new_tokens: 生成する最大トークン数

        Returns:
            生成された説明文
        """
        pil_image = Image.fromarray(image)

        if mask is not None:
            pil_image = self._crop_with_mask(image, mask)
            if pil_image is None:
                return "対象物が見つかりませんでした"

        if prompt:
            inputs = self.processor(
                images=pil_image, text=prompt, return_tensors="pt"
            ).to(self.device, self.dtype)
        else:
            inputs = self.processor(images=pil_image, return_tensors="pt").to(
                self.device, self.dtype
            )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return caption.strip()

    def generate_sound_description(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> str:
        """対象物の音に関する説明文を生成（SAM Audio用）

        Args:
            image: RGB形式の画像 (H, W, 3)
            mask: オプションのセグメンテーションマスク (H, W)

        Returns:
            音声分離に適した説明文
        """
        # まず対象物が何かを特定
        what_prompt = "Question: What is this object? Answer:"
        object_name = self.generate_caption(image, mask, prompt=what_prompt)

        # 音に関する説明を生成
        sound_prompt = f"Question: What sound does {object_name} make? Answer:"
        sound_description = self.generate_caption(image, mask, prompt=sound_prompt)

        # SAM Audio用に適切なフォーマットに整形
        # 一般的なフォーマット: "A [object] [making sound]"
        if sound_description and object_name:
            return f"{object_name} {sound_description}"
        elif object_name:
            return object_name

        # フォールバック: 単純なキャプション
        return self.generate_caption(image, mask)

    def _crop_with_mask(
        self, image: np.ndarray, mask: np.ndarray, padding: int = 20
    ) -> Optional[Image.Image]:
        """マスク領域を切り出す

        Args:
            image: RGB形式の画像 (H, W, 3)
            mask: セグメンテーションマスク (H, W)
            padding: パディング（ピクセル）

        Returns:
            切り出したPIL画像、またはNone
        """
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        y1, y2 = y_indices[0], y_indices[-1]
        x1, x2 = x_indices[0], x_indices[-1]

        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        cropped = image[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    def unload(self):
        """モデルをアンロードしてメモリを解放"""
        self._model = None
        self._processor = None
        from ..config.device import clear_memory

        clear_memory(self.device)
