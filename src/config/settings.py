from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AppSettings:
    """アプリケーション設定"""

    # モデル設定
    sam3_model: str = "facebook/sam3"
    blip2_model: str = "Salesforce/blip2-opt-2.7b"
    sam_audio_model: str = "facebook/sam-audio-small"

    # パス設定
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    model_cache_dir: Path = field(default_factory=lambda: Path("models"))

    # 処理設定
    max_frames: int = 100
    frame_sample_rate: int = 5

    # UI設定
    server_port: int = 7860
    share: bool = False

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)
        self.model_cache_dir.mkdir(exist_ok=True)
