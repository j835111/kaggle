"""事後校準：temperature scaling。

log loss 吃的是機率校準品質，不是準確率——不用重新訓練，只要在驗證集的 logits 上
配一個純量溫度 T，讓 softmax(logits / T) 更校準，套用到測試集的 logits 上即可。

純 numpy，不需要 torch，可以在本機測試（真正的 logits 由 llmcls/train.py 的
predict_logits() / predict_logits_with_model() 產生，那兩個需要 torch/transformers）。
"""

from __future__ import annotations

import warnings

import numpy as np

from llmcls.metrics import log_loss, softmax


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """回傳 softmax(logits / temperature)。temperature > 1 會讓機率分佈變平滑
    （撫平過度自信），temperature < 1 會讓分佈更尖銳，temperature == 1 等於沒校準。
    """
    return softmax(np.asarray(logits) / temperature)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    lo: float = 0.1,
    hi: float = 5.0,
    n_grid: int = 50,
    n_refine: int = 4,
) -> float:
    """在 [lo, hi] 網格搜尋、逐步細化，找出讓 log loss 最小的溫度 T。

    T 只有一個純量，網格搜尋 + 逐步細化就足夠穩定，不需要另外拉 scipy 依賴
    （sklearn 有牽帶到 scipy，但那是 transitive dependency，不想仰賴它）。
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    orig_lo, orig_hi = lo, hi
    best_t = 1.0
    cur_lo, cur_hi = lo, hi
    for _ in range(n_refine):
        candidates = np.linspace(cur_lo, cur_hi, n_grid)
        losses = [log_loss(labels, apply_temperature(logits, t)) for t in candidates]
        idx = int(np.argmin(losses))
        best_t = float(candidates[idx])
        span = (cur_hi - cur_lo) / n_grid
        cur_lo, cur_hi = max(1e-3, best_t - span), best_t + span

    if best_t <= orig_lo * 1.05 or best_t >= orig_hi * 0.95:
        warnings.warn(
            f"fit_temperature 找到的 T={best_t:.3f} 貼著搜尋邊界 [{orig_lo}, {orig_hi}]，"
            "可能沒收斂，先擴大 lo/hi 範圍再看一次"
        )
    return best_t
