"""訓練中途的「要不要緊急煞車」判斷邏輯，抽成不依賴 torch/transformers 的純函式
——可以在本機測試決策本身對不對，不用等 GPU 上真的發散一次才知道邏輯有沒有錯。

真正接到 HF Trainer 的地方在 llmcls/train.py 的 StopOnNonFiniteLoss。
"""

from __future__ import annotations

import math


def should_stop_for_nonfinite(
    logs: dict,
    consecutive_nonfinite_grad_norm: int,
    grad_norm_patience: int = 3,
) -> tuple[bool, int]:
    """回傳 (要不要喊停, 更新後的連續非有限 grad_norm 次數)。

    `loss` 本身變成非有限值代表訓練真的壞了（權重已經被非有限值污染，之後每一步
    都是壞的），一次就立刻停。`grad_norm` 偶爾出現非有限值不是同一回事——fp16
    混合精度訓練下，GradScaler 算出某一步的梯度真的溢位時會自動跳過那次更新、
    調低 scale factor 再繼續，這是設計上就會發生的正常現象，loss 本身仍然健康，
    不代表模型壞掉。實測踩過這個誤判：group_by_length 那次「發散」在觸發停止的
    那一行 loss 是 1.079（跟前面每一步一樣正常），只有 grad_norm 是 inf——判定
    「有害」其實是這個過度敏感的偵測邏輯誤觸發，不是真的訓練壞掉。改成只有連續
    `grad_norm_patience` 次 log 都非有限值（不是單一次）才當作真的卡住、喊停。
    """
    loss = logs.get("loss")
    if isinstance(loss, (int, float)) and not math.isfinite(loss):
        return True, consecutive_nonfinite_grad_norm

    grad_norm = logs.get("grad_norm")
    if isinstance(grad_norm, (int, float)) and not math.isfinite(grad_norm):
        consecutive_nonfinite_grad_norm += 1
    else:
        consecutive_nonfinite_grad_norm = 0

    return consecutive_nonfinite_grad_norm >= grad_norm_patience, consecutive_nonfinite_grad_norm
