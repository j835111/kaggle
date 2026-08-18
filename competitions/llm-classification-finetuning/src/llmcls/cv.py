"""交叉驗證切分。

重點：訓練集裡有數千筆重複的 prompt。如果同一個 prompt 同時落在 train 和 valid
fold，本地分數會虛高、跟 LB 對不上。所以一律以 prompt 當 group 切分。
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from llmcls.config import N_FOLDS, SEED


def prompt_group_key(df: pd.DataFrame) -> pd.Series:
    """以 prompt 文字的 hash 當 group id（正規化空白後再 hash）。"""
    normalized = df["prompt_text"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
    return normalized.map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())


def add_folds(
    df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED, col: str = "fold"
) -> pd.DataFrame:
    """加上 fold 欄位：以 prompt 分組、以 label 分層。回傳新的 DataFrame。"""
    df = df.copy()
    groups = prompt_group_key(df)
    n_groups = groups.nunique()
    if n_groups < n_folds:
        raise ValueError(f"唯一 prompt 數 ({n_groups}) 少於 fold 數 ({n_folds})，無法切分")

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df[col] = -1
    for fold, (_, valid_idx) in enumerate(splitter.split(df, df["label"], groups)):
        df.iloc[valid_idx, df.columns.get_loc(col)] = fold

    assert (df[col] >= 0).all(), "有列沒有被分配到 fold"
    _assert_no_group_leak(df, groups, col)
    return df


def _assert_no_group_leak(df: pd.DataFrame, groups: pd.Series, col: str) -> None:
    """驗證每個 prompt group 只出現在單一 fold —— 這是切分的重點，值得直接斷言。"""
    per_group = pd.DataFrame({"group": groups.to_numpy(), "fold": df[col].to_numpy()})
    leaked = per_group.groupby("group")["fold"].nunique()
    n_leaked = int((leaked > 1).sum())
    if n_leaked:
        raise AssertionError(f"有 {n_leaked} 個 prompt group 跨越多個 fold，切分有誤")


def fold_indices(df: pd.DataFrame, fold: int, col: str = "fold") -> tuple[np.ndarray, np.ndarray]:
    """回傳 (train_idx, valid_idx) 的位置索引。"""
    is_valid = (df[col] == fold).to_numpy()
    return np.flatnonzero(~is_valid), np.flatnonzero(is_valid)
