"""シード固定機能のテスト"""

import torch
import numpy as np
import random
import pytest


class TestSetSeed:
    """set_seed関数のテスト"""

    def test_set_seed_exists(self):
        """set_seed関数がインポートできる"""
        from src.audio.sam_audio_wrapper import set_seed

        assert callable(set_seed)

    def test_torch_reproducibility(self):
        """同じシードで同じtorch乱数が生成される"""
        from src.audio.sam_audio_wrapper import set_seed

        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(42)
        tensor2 = torch.randn(10)

        assert torch.allclose(tensor1, tensor2)

    def test_numpy_reproducibility(self):
        """同じシードで同じnumpy乱数が生成される"""
        from src.audio.sam_audio_wrapper import set_seed

        set_seed(42)
        arr1 = np.random.rand(10)

        set_seed(42)
        arr2 = np.random.rand(10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_random_reproducibility(self):
        """同じシードで同じrandom乱数が生成される"""
        from src.audio.sam_audio_wrapper import set_seed

        set_seed(42)
        vals1 = [random.random() for _ in range(10)]

        set_seed(42)
        vals2 = [random.random() for _ in range(10)]

        assert vals1 == vals2

    def test_different_seeds_produce_different_results(self):
        """異なるシードで異なる乱数が生成される"""
        from src.audio.sam_audio_wrapper import set_seed

        set_seed(42)
        tensor1 = torch.randn(10)

        set_seed(123)
        tensor2 = torch.randn(10)

        assert not torch.allclose(tensor1, tensor2)


class TestSAMAudioWrapperSeed:
    """SAMAudioWrapperのシード機能テスト"""

    def test_wrapper_accepts_seed_parameter(self):
        """SAMAudioWrapperがseedパラメータを受け取れる"""
        from src.audio.sam_audio_wrapper import SAMAudioWrapper

        wrapper = SAMAudioWrapper(seed=42)
        assert wrapper.seed == 42

    def test_wrapper_seed_default_is_none(self):
        """seedのデフォルト値はNone"""
        from src.audio.sam_audio_wrapper import SAMAudioWrapper

        wrapper = SAMAudioWrapper()
        assert wrapper.seed is None
