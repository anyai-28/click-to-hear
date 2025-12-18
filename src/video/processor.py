import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional
from moviepy import VideoFileClip


class VideoProcessor:
    """動画処理クラス"""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """特定フレームを取得 (RGB形式)"""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"フレームインデックスが範囲外です: {frame_idx}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"フレーム {frame_idx} を読み込めませんでした")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def extract_frames(
        self, sample_rate: int = 1, max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """フレームをジェネレータとして抽出

        Args:
            sample_rate: サンプリング間隔 (例: 5なら5フレームごと)
            max_frames: 最大フレーム数

        Yields:
            (フレームインデックス, RGB形式のフレーム画像)
        """
        count = 0
        for i in range(0, self.frame_count, sample_rate):
            if max_frames and count >= max_frames:
                break
            try:
                yield i, self.get_frame(i)
                count += 1
            except ValueError:
                continue

    def extract_audio(self, output_path: str) -> str:
        """動画から音声を抽出してWAVファイルとして保存

        Args:
            output_path: 出力ファイルパス (.wav)

        Returns:
            保存したファイルパス
        """
        video = VideoFileClip(str(self.video_path))

        if video.audio is None:
            raise ValueError("この動画には音声トラックがありません")

        video.audio.write_audiofile(output_path, logger=None)
        video.close()

        return output_path

    def get_info(self) -> dict:
        """動画情報を取得"""
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
        }

    def close(self):
        """リソースを解放"""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
