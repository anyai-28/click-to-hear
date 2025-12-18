import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List

from ..config.device import get_device_for_model, get_dtype_for_device


class SAM3Wrapper:
    """SAM3 セグメンテーションラッパー"""

    def __init__(self, model_name: str = "facebook/sam3"):
        self.model_name = model_name
        self.device = get_device_for_model("sam3")
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
        import os
        token = os.environ.get("HF_TOKEN")

        try:
            # transformers版を試行（ポイント/ボックス/テキスト対応のTracker版を使用）
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            self._processor = Sam3TrackerProcessor.from_pretrained(self.model_name, token=token)
            self._model = Sam3TrackerModel.from_pretrained(
                self.model_name, torch_dtype=self.dtype, token=token
            ).to(self.device)
        except ImportError:
            # 公式リポジトリ版にフォールバック
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self._model = build_sam3_image_model()
            self._processor = Sam3Processor(self._model)

        self._model.eval()

    def segment_with_point(
        self, image: np.ndarray, point: Tuple[int, int], label: int = 1
    ) -> np.ndarray:
        """ポイントクリックでセグメンテーション

        Args:
            image: RGB形式の画像 (H, W, 3)
            point: クリック座標 (x, y)
            label: 1=ポジティブ（対象）, 0=ネガティブ（背景）

        Returns:
            セグメンテーションマスク (H, W), 0-1の値
        """
        pil_image = Image.fromarray(image)

        try:
            # transformers版のAPI（Sam3TrackerModel/Sam3TrackerProcessor）
            input_points = [[[[point[0], point[1]]]]]
            input_labels = [[[label]]]

            inputs = self.processor(
                images=pil_image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]

            # iou_scoresを使って最も信頼度の高いマスクを選択
            best_mask_idx = outputs.iou_scores.squeeze().argmax().item()
            return masks[0, best_mask_idx].numpy().astype(np.float32)

        except AttributeError:
            # 公式リポジトリ版のAPI
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_point_prompts(
                state=inference_state, points=[[point[0], point[1]]], labels=[label]
            )
            return output["masks"][0].numpy().astype(np.float32)

    def segment_with_text(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        """テキストプロンプトでセグメンテーション

        Args:
            image: RGB形式の画像 (H, W, 3)
            text_prompt: セグメント対象を指定するテキスト

        Returns:
            セグメンテーションマスク (H, W), 0-1の値

        Note:
            Sam3TrackerModelはテキストプロンプトをサポートしていない可能性があります。
            テキストベースのセグメンテーションが必要な場合は、
            Sam3Model/Sam3Processorを使用する必要があるかもしれません。
        """
        pil_image = Image.fromarray(image)

        try:
            # transformers版のAPI
            inputs = self.processor(
                images=pil_image, text=text_prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]

            return masks[0, 0].numpy().astype(np.float32)

        except AttributeError:
            # 公式リポジトリ版のAPI
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(
                state=inference_state, prompt=text_prompt
            )
            return output["masks"][0].numpy().astype(np.float32)

    def segment_with_box(
        self, image: np.ndarray, box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """バウンディングボックスでセグメンテーション

        Args:
            image: RGB形式の画像 (H, W, 3)
            box: (x1, y1, x2, y2) 形式のバウンディングボックス

        Returns:
            セグメンテーションマスク (H, W), 0-1の値
        """
        pil_image = Image.fromarray(image)

        try:
            # transformers版のAPI（Sam3TrackerModel/Sam3TrackerProcessor）
            # input_boxesは3次元: [[[x1, y1, x2, y2]]]
            inputs = self.processor(
                images=pil_image,
                input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]

            return masks[0, 0].numpy().astype(np.float32)

        except AttributeError:
            # 公式リポジトリ版のAPI
            inference_state = self.processor.set_image(pil_image)
            output = self.processor.set_box_prompt(state=inference_state, box=box)
            return output["masks"][0].numpy().astype(np.float32)

    def unload(self):
        """モデルをアンロードしてメモリを解放"""
        self._model = None
        self._processor = None
        from ..config.device import clear_memory

        clear_memory(self.device)


def visualize_mask(
    image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """マスクを画像にオーバーレイ表示

    Args:
        image: RGB形式の画像 (H, W, 3)
        mask: セグメンテーションマスク (H, W), 0-1の値
        color: オーバーレイの色 (R, G, B)

    Returns:
        オーバーレイ画像 (H, W, 3)
    """
    overlay = image.copy()
    mask_binary = mask > 0.5

    # マスク領域に半透明の色を重ねる
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask_binary] = color

    alpha = 0.4
    overlay = (overlay * (1 - alpha * mask_binary[..., np.newaxis])).astype(np.uint8)
    overlay = overlay + (mask_rgb * alpha).astype(np.uint8)

    # マスクの輪郭を描画
    import cv2

    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


def get_mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """マスクからバウンディングボックスを取得

    Args:
        mask: セグメンテーションマスク (H, W)

    Returns:
        (x1, y1, x2, y2) または None（マスクが空の場合）
    """
    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    return (
        int(x_indices[0]),
        int(y_indices[0]),
        int(x_indices[-1]),
        int(y_indices[-1]),
    )


def crop_with_mask(
    image: np.ndarray, mask: np.ndarray, padding: int = 10
) -> Optional[np.ndarray]:
    """マスク領域を切り出す

    Args:
        image: RGB形式の画像 (H, W, 3)
        mask: セグメンテーションマスク (H, W)
        padding: パディング（ピクセル）

    Returns:
        切り出した画像、またはNone（マスクが空の場合）
    """
    bbox = get_mask_bbox(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # パディングを追加
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[y1:y2, x1:x2].copy()
