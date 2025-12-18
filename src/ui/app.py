import gradio as gr
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from ..config.settings import AppSettings
from ..config.device import get_device_for_model
from ..video.processor import VideoProcessor
from ..segmentation.sam3_wrapper import SAM3Wrapper, visualize_mask
from ..captioning.blip2_captioner import BLIP2Captioner
from ..audio.sam_audio_wrapper import SAMAudioWrapper, save_audio


class SAMApp:
    """SAM3 + SAM Audio 統合アプリケーション"""

    def __init__(self, settings: AppSettings):
        self.settings = settings

        # モデル（遅延ロード）
        self._sam3: Optional[SAM3Wrapper] = None
        self._captioner: Optional[BLIP2Captioner] = None
        self._sam_audio: Optional[SAMAudioWrapper] = None

        # 状態
        self.current_video_path: Optional[str] = None
        self.current_processor: Optional[VideoProcessor] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_idx: int = 0
        self.current_mask: Optional[np.ndarray] = None
        self.current_caption: str = ""

    @property
    def sam3(self) -> SAM3Wrapper:
        if self._sam3 is None:
            self._sam3 = SAM3Wrapper(self.settings.sam3_model)
        return self._sam3

    @property
    def captioner(self) -> BLIP2Captioner:
        if self._captioner is None:
            self._captioner = BLIP2Captioner(self.settings.blip2_model)
        return self._captioner

    @property
    def sam_audio(self) -> SAMAudioWrapper:
        if self._sam_audio is None:
            self._sam_audio = SAMAudioWrapper(self.settings.sam_audio_model)
        return self._sam_audio

    def on_video_upload(self, video_path: str) -> Tuple[gr.Slider, np.ndarray, str]:
        """動画アップロード時の処理"""
        if video_path is None:
            return gr.Slider(maximum=0, value=0), None, ""

        # 前の動画を閉じる
        if self.current_processor is not None:
            self.current_processor.close()

        self.current_video_path = video_path
        self.current_processor = VideoProcessor(video_path)
        self.current_frame_idx = 0
        self.current_frame = self.current_processor.get_frame(0)
        self.current_mask = None
        self.current_caption = ""

        info = self.current_processor.get_info()
        info_text = f"サイズ: {info['width']}x{info['height']}, FPS: {info['fps']:.1f}, 長さ: {info['duration']:.1f}秒"

        return (
            gr.Slider(
                minimum=0,
                maximum=self.current_processor.frame_count - 1,
                value=0,
                step=1,
            ),
            self.current_frame,
            info_text,
        )

    def on_frame_change(self, frame_idx: int) -> np.ndarray:
        """フレームスライダー変更時"""
        if self.current_processor is None:
            return None

        self.current_frame_idx = int(frame_idx)
        self.current_frame = self.current_processor.get_frame(self.current_frame_idx)
        self.current_mask = None
        self.current_caption = ""

        return self.current_frame

    def on_image_click(
        self, evt: gr.SelectData
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """画像クリック時のセグメンテーション"""
        if self.current_frame is None:
            return None, None, ""

        x, y = evt.index
        print(f"クリック座標: ({x}, {y})")

        # セグメンテーション実行
        self.current_mask = self.sam3.segment_with_point(
            self.current_frame, (x, y), label=1
        )

        # マスク可視化
        overlay = visualize_mask(self.current_frame, self.current_mask, (0, 255, 0))

        # 説明文生成
        self.current_caption = self.captioner.generate_caption(
            self.current_frame, self.current_mask
        )

        return self.current_frame, overlay, self.current_caption

    def on_caption_edit(self, new_caption: str) -> str:
        """説明文の編集"""
        self.current_caption = new_caption
        return new_caption

    def on_separate(self, caption: str) -> Tuple[Optional[str], Optional[str], str]:
        """音声分離実行"""
        if self.current_processor is None:
            return None, None, "動画をアップロードしてください"

        if not caption:
            return None, None, "説明文を入力してください"

        try:
            # 音声抽出
            audio_path = str(self.settings.output_dir / "temp_audio.wav")
            self.current_processor.extract_audio(audio_path)

            # SAM Audioで分離
            result = self.sam_audio.separate_by_text(audio_path, caption)

            # 保存
            target_path = str(self.settings.output_dir / "target.wav")
            residual_path = str(self.settings.output_dir / "residual.wav")

            save_audio(result.target, target_path, result.sample_rate)
            save_audio(result.residual, residual_path, result.sample_rate)

            return target_path, residual_path, "音声分離が完了しました"

        except Exception as e:
            return None, None, f"エラー: {str(e)}"

    def build_ui(self) -> gr.Blocks:
        """Gradio UIを構築"""
        with gr.Blocks(
            title="SAM Video Audio Separator",
            theme=gr.themes.Soft(),
        ) as demo:
            gr.Markdown(
                """
                # SAM3 + SAM Audio: 動画セグメンテーション & 音声分離

                1. 動画をアップロード
                2. フレームスライダーで表示フレームを選択
                3. 画像をクリックして対象物をセグメント
                4. 説明文を確認・編集
                5. 「音声分離実行」をクリック
                """
            )

            with gr.Row():
                # 左カラム: 入力
                with gr.Column(scale=1):
                    video_input = gr.Video(label="動画をアップロード")
                    video_info = gr.Textbox(label="動画情報", interactive=False)
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=0,
                        label="フレーム選択",
                    )
                    current_frame_display = gr.Image(
                        label="現在のフレーム（クリックでセグメント）",
                        interactive=True,
                        type="numpy",
                    )

                # 右カラム: 出力
                with gr.Column(scale=1):
                    segmented_image = gr.Image(
                        label="セグメンテーション結果", type="numpy"
                    )
                    caption_output = gr.Textbox(
                        label="説明文（編集可能）",
                        interactive=True,
                        placeholder="クリックして対象物を選択すると、説明文が生成されます",
                    )

                    with gr.Row():
                        separate_btn = gr.Button(
                            "音声分離実行", variant="primary", scale=2
                        )

                    status_text = gr.Textbox(label="ステータス", interactive=False)

                    gr.Markdown("### 分離された音声")
                    target_audio = gr.Audio(
                        label="対象音声（抽出）", type="filepath", interactive=False
                    )
                    residual_audio = gr.Audio(
                        label="背景音声（除去）", type="filepath", interactive=False
                    )

            # イベントハンドラー
            video_input.change(
                self.on_video_upload,
                inputs=[video_input],
                outputs=[frame_slider, current_frame_display, video_info],
            )

            frame_slider.change(
                self.on_frame_change,
                inputs=[frame_slider],
                outputs=[current_frame_display],
            )

            current_frame_display.select(
                self.on_image_click,
                inputs=[],
                outputs=[current_frame_display, segmented_image, caption_output],
            )

            caption_output.change(
                self.on_caption_edit,
                inputs=[caption_output],
                outputs=[caption_output],
            )

            separate_btn.click(
                self.on_separate,
                inputs=[caption_output],
                outputs=[target_audio, residual_audio, status_text],
            )

        return demo

    def launch(self, **kwargs):
        """アプリを起動"""
        demo = self.build_ui()
        demo.launch(
            server_port=self.settings.server_port,
            share=self.settings.share,
            **kwargs,
        )


def create_app(settings: Optional[AppSettings] = None) -> SAMApp:
    """アプリケーションインスタンスを作成"""
    if settings is None:
        settings = AppSettings()
    return SAMApp(settings)
