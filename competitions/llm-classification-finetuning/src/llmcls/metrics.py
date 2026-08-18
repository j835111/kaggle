"""評分指標。競賽用 multi-class log loss。"""

from __future__ import annotations

import numpy as np

from llmcls.config import N_CLASSES

# 均勻亂猜的分數 = ln(3) ≈ 1.0986。任何模型沒打敗這條線就等於沒有資訊量。
UNIFORM_LOGLOSS = float(np.log(N_CLASSES))

EPS = 1e-15


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = EPS) -> float:
    """multi-class log loss。y_true 是整數 label，y_prob 是 (n, n_classes) 機率。

    先 clip 到 [eps, 1-eps] 再逐列重新正規化。Kaggle 端的實際實作未經查證，
    但只要機率沒有極端到觸及 eps，各種變體的差異可忽略。
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2 or y_prob.shape[1] != N_CLASSES:
        raise ValueError(f"y_prob 形狀應為 (n, {N_CLASSES})，實際為 {y_prob.shape}")
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true ({len(y_true)}) 與 y_prob ({len(y_prob)}) 長度不一致")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob 含有 NaN 或 inf")

    p = np.clip(y_prob, eps, 1 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))
