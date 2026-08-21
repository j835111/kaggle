"""a/b 對調 TTA（test-time augmentation）：對付位置偏誤。

模型可能學到偏好某個位置（A 或 B）本身，而不是純粹依回覆品質判斷。把 response_a /
response_b 對調後再推論一次，兩次結果換回原本的欄位對齊後平均，讓最終預測不受
位置影響。

純 numpy，不需要 torch，可以在本機測試（真正的兩次推論在 llmcls/train.py，
需要 torch/transformers）。
"""

from __future__ import annotations

import numpy as np

# LABEL_COLS 的欄位順序：[winner_model_a, winner_model_b, winner_tie]。
_SWAP_COLUMNS = [1, 0, 2]


def average_swapped(arr_orig: np.ndarray, arr_swapped: np.ndarray) -> np.ndarray:
    """把對調順序推論出來的結果換回原本的 a/b 欄位對齊，再跟原始順序的結果平均。

    `arr_swapped` 是把 response_a/response_b 對調後推論出來的，它的欄位順序是
    [P(目前放在 A 位置的贏), P(目前放在 B 位置的贏), P(tie)]——但「目前放在 A 位置」
    的其實是原本的 response_b，所以要先把前兩欄位換回來（`arr_swapped[:, [1, 0, 2]]`），
    才能跟 `arr_orig` 對齊平均。同一個操作對 logits（softmax 之前）跟機率（softmax
    之後）都適用，純粹是欄位重排 + 逐元素平均。
    """
    arr_orig = np.asarray(arr_orig)
    arr_swapped = np.asarray(arr_swapped)
    if arr_orig.shape != arr_swapped.shape:
        raise ValueError(f"形狀不一致：arr_orig {arr_orig.shape} vs arr_swapped {arr_swapped.shape}")
    aligned = arr_swapped[:, _SWAP_COLUMNS]
    return (arr_orig + aligned) / 2
