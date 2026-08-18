"""提交檔組裝與驗證。

Kaggle 只會告訴你「submission 格式錯誤」，不會告訴你錯在哪裡，
所以在本機就把能檢查的都檢查掉。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llmcls.config import LABEL_COLS, N_CLASSES, OUTPUT_DIR, SAMPLE_SUBMISSION_CSV


def build_submission(ids: pd.Series | np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise ValueError(f"probs 形狀應為 (n, {N_CLASSES})，實際為 {probs.shape}")
    if len(ids) != len(probs):
        raise ValueError(f"ids ({len(ids)}) 與 probs ({len(probs)}) 長度不一致")
    sub = pd.DataFrame({"id": np.asarray(ids)})
    sub[LABEL_COLS] = probs
    return sub


def validate_submission(sub: pd.DataFrame, sample_path: Path | None = None) -> None:
    """檢查欄位、數值範圍、機率和；有 sample_submission 時再比對 id 集合。"""
    expected_cols = ["id", *LABEL_COLS]
    if list(sub.columns) != expected_cols:
        raise ValueError(f"欄位應為 {expected_cols}，實際為 {list(sub.columns)}")

    probs = sub[LABEL_COLS].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ValueError("提交檔含有 NaN 或 inf")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("機率值超出 [0, 1] 範圍")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        worst = float(np.abs(row_sums - 1.0).max())
        raise ValueError(f"每列機率和必須為 1，最大偏差 {worst:.2e}")
    if sub["id"].duplicated().any():
        raise ValueError("提交檔有重複的 id")

    sample_path = Path(sample_path) if sample_path is not None else SAMPLE_SUBMISSION_CSV
    if sample_path.exists():
        expected_ids = set(pd.read_csv(sample_path)["id"])
        actual_ids = set(sub["id"])
        if expected_ids != actual_ids:
            raise ValueError(
                f"id 集合與 sample_submission 不符："
                f"缺少 {len(expected_ids - actual_ids)} 筆、多出 {len(actual_ids - expected_ids)} 筆"
            )


def save_submission(
    sub: pd.DataFrame, name: str = "submission.csv", sample_path: Path | None = None
) -> Path:
    validate_submission(sub, sample_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    sub.to_csv(path, index=False)
    return path
