"""資料載入與欄位解析。

注意：prompt / response_a / response_b 在原始 CSV 裡是 **JSON 編碼的字串陣列**
（多輪對話，每個 element 是一輪），不是純文字。這點在真實資料下載前尚未實地驗證，
所以 parse 採防禦式寫法：json.loads 失敗就退回當成單輪純文字。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from llmcls.config import LABEL_COLS, TEST_CSV, TEXT_COLS, TRAIN_CSV

TURN_SEP = "\n\n"

# 退回純文字的比例超過這個門檻就直接報錯 —— 代表該欄根本不是 JSON，schema 假設錯了。
FALLBACK_ERROR_RATIO = 0.01


def _is_missing(raw: object) -> bool:
    """型別無關的缺值判斷（None / float nan / pd.NA 都算）。"""
    if raw is None:
        return True
    try:
        return bool(pd.isna(raw))
    except (TypeError, ValueError):
        # 例如 list、ndarray：pd.isna 回傳陣列或直接拋錯，都當作非缺值。
        return False


def _parse_turns_flagged(raw: object) -> tuple[list[str], bool]:
    """回傳 (turns, 是否退回純文字)。退回的次數會被上層統計，不能靜默吞掉。"""
    if _is_missing(raw):
        return [], False
    if isinstance(raw, list):
        return [("" if t is None else str(t)) for t in raw], False
    text = str(raw)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 不是合法 JSON —— 當成單輪純文字，避免整批資料因為個別壞格式而中斷。
        return [text], True
    if isinstance(parsed, list):
        return [("" if t is None else str(t)) for t in parsed], False
    return [str(parsed)], True


def parse_turns(raw: object) -> list[str]:
    """把一格 JSON 字串解析成 list[str]；缺值 / 解析失敗都不會炸掉。"""
    return _parse_turns_flagged(raw)[0]


def join_turns(turns: list[str]) -> str:
    return TURN_SEP.join(t for t in turns if t)


def _require_columns(df: pd.DataFrame, cols: list[str], source: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} 缺少預期欄位 {missing}；實際欄位為 {list(df.columns)}。"
            " 若官方 schema 有變動，請同步更新 llmcls/config.py。"
        )


def _add_parsed_text(df: pd.DataFrame) -> pd.DataFrame:
    """解析文字欄位，並統計有多少列走了「退回純文字」的退路。

    這個計數是 schema 假設是否成立的第一手診斷：少數幾列是雜訊，比例一高就代表
    該欄根本不是 JSON。絕對不能靜默退回，否則模型會拿著原始 JSON 字串在訓練。
    """
    for col in TEXT_COLS:
        flagged = df[col].map(_parse_turns_flagged)
        turns = flagged.map(lambda x: x[0])
        n_fallback = int(flagged.map(lambda x: x[1]).sum())
        if n_fallback:
            msg = f"{col}：{n_fallback}/{len(df)} 列無法解析為 JSON 陣列，已退回單輪純文字"
            if n_fallback > FALLBACK_ERROR_RATIO * len(df):
                raise ValueError(
                    f"{msg} —— 比例過高，該欄的 schema 可能與預期不符。"
                    " 請檢查實際資料格式並更新 llmcls/data.py 的解析邏輯。"
                )
            warnings.warn(msg, stacklevel=2)
        df[f"{col}_turns"] = turns
        df[f"{col}_text"] = turns.map(join_turns)
    return df


def load_train(path: Path | None = None) -> pd.DataFrame:
    """載入訓練集，附上解析後的文字欄位與整數 label。"""
    path = Path(path) if path is not None else TRAIN_CSV
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}；請先執行 scripts/download_data.sh 下載競賽資料。")
    df = pd.read_csv(path)
    _require_columns(df, ["id", *TEXT_COLS, *LABEL_COLS], path)

    onehot = df[LABEL_COLS].to_numpy()
    bad = onehot.sum(axis=1) != 1
    if bad.any():
        raise ValueError(
            f"{path} 有 {int(bad.sum())} 列的 {LABEL_COLS} 不是恰好一個 1，"
            " 無法轉成單一 label，請檢查資料。"
        )
    df["label"] = onehot.argmax(axis=1)
    return _add_parsed_text(df)


def load_test(path: Path | None = None) -> pd.DataFrame:
    """載入測試集（沒有 label 欄）。"""
    path = Path(path) if path is not None else TEST_CSV
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}；請先執行 scripts/download_data.sh 下載競賽資料。")
    df = pd.read_csv(path)
    _require_columns(df, ["id", *TEXT_COLS], path)
    return _add_parsed_text(df)
