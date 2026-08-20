"""貼進 Kaggle Notebook 第一個 cell —— 由 scripts/gen_notebook_bootstrap.py 產生，不要手改。"""

import pathlib
import sys

_llmcls_files = {
    '__init__.py': r'''"""LLM Classification Finetuning — 共用工具模組。

刻意寫成可 import 的模組（而不是單一 notebook），因為訓練會在
Kaggle Notebook 或遠端 GPU 上跑，本機只負責資料處理、CV 切分與提交組裝。
同一份程式碼兩邊都能 import，路徑差異由 config.py 吸收。
"""

from llmcls.config import DATA_DIR, LABEL_COLS, OUTPUT_DIR
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss

__all__ = ["DATA_DIR", "OUTPUT_DIR", "LABEL_COLS", "log_loss", "UNIFORM_LOGLOSS"]
''',
    'config.py': r'''"""路徑與常數。本機 / Kaggle Notebook 的差異全部集中在這裡。"""

from __future__ import annotations

import os
from pathlib import Path

COMPETITION = "llm-classification-finetuning"

# Kaggle Notebook 內資料掛載路徑：透過網頁 UI「Add Data」掛的話是
# /kaggle/input/<competition>/，但實測透過 `kaggle kernels push`（kernel-metadata.json
# 的 competition_sources）掛的話，實際掛在 /kaggle/input/competitions/<competition>/，
# 多一層 competitions/ —— 兩條路徑都要認，不要假設只有一種。
# 可用環境變數 LLMCLS_DATA_DIR 覆寫（例如指到 data/fixture 跑煙霧測試）。
_KAGGLE_INPUT_CANDIDATES = [
    Path("/kaggle/input") / COMPETITION,
    Path("/kaggle/input/competitions") / COMPETITION,
]
# 本競賽目錄 competitions/<slug>/，不是 git repo 根目錄 —— 工作區還有其他競賽。
_COMP_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_dir() -> Path:
    if env := os.environ.get("LLMCLS_DATA_DIR"):
        return Path(env)
    for candidate in _KAGGLE_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return _COMP_ROOT / "data"


DATA_DIR = _resolve_data_dir()
# Kaggle Notebook 只有 /kaggle/working 可寫。
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else _COMP_ROOT / "outputs"

# 訓練與推論是兩個獨立的 Kaggle Notebook：訓練 notebook 把權重存到 OUTPUT_DIR/model，
# Save Version 後那個資料夾變成一個 Kaggle Dataset，掛進推論 notebook 時的掛載路徑
# 由使用者在 Kaggle UI 上決定、無法預先得知，所以用環境變數覆寫（呼應 LLMCLS_DATA_DIR）。
MODEL_DIR = Path(os.environ.get("LLMCLS_MODEL_DIR", str(OUTPUT_DIR / "model")))

# 訓練用的 base model；推論 notebook 離線，權重從 MODEL_DIR 讀，不會連到這個 hub id。
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 512

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# 三分類的目標欄位，順序即 class 0/1/2，全專案共用這個順序。
LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]
N_CLASSES = len(LABEL_COLS)

# 需要 parse 的 JSON 字串欄位（多輪對話存成 list of str）。
TEXT_COLS = ["prompt", "response_a", "response_b"]

SEED = 42
N_FOLDS = 5
''',
    'cv.py': r'''"""交叉驗證切分。

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
''',
    'data.py': r'''"""資料載入與欄位解析。

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
''',
    'metrics.py': r'''"""評分指標。競賽用 multi-class log loss。"""

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
''',
    'submission.py': r'''"""提交檔組裝與驗證。

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
''',
    'text.py': r'''"""把 prompt / response_a / response_b 的 token id 組成單一模型輸入序列，
超過 max_len 時做 head+tail 截斷。

刻意在 id 層級操作（呼叫端先分別 encode 三段，這裡只管截斷與長度預算），而不是
組字串再整段重新 tokenize —— 這樣三段可以各自截斷，不會因為 response_a 太長就把
response_b 擠到只剩尾巴一小段，兩邊被模型看到的資訊量比較公平。

不依賴 torch / transformers，純 Python 可在本機測試。真正呼叫 tokenizer 的地方在
llmcls/train.py（那裡才需要 GPU 環境）。
"""

from __future__ import annotations

# 對應 [CLS] prompt [SEP] response_a [SEP] response_b [SEP] 的組法：1 個 CLS + 3 個 SEP。
NUM_SPECIAL_TOKENS = 4

# 三段的預算比例：prompt 通常比兩份回覆短，兩份回覆同等重要所以各拿更多。
DEFAULT_RATIOS = (0.2, 0.4, 0.4)


def truncate_ids(ids: list[int], budget: int, head_ratio: float = 0.5) -> list[int]:
    """留頭尾、砍中間。budget <= 0 回傳空列表，不需要截斷時原樣回傳。"""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    head_len = min(budget, max(1, round(budget * head_ratio)))
    tail_len = budget - head_len
    if tail_len <= 0:
        return ids[:head_len]
    return ids[:head_len] + ids[len(ids) - tail_len :]


def split_budget(total: int, ratios: tuple[float, float, float] = DEFAULT_RATIOS) -> tuple[int, int, int]:
    """依 ratios 把 total 分給三段；四捨五入的誤差全部歸給最後一段，確保三段總和精確等於 total。"""
    if total <= 0:
        return (0, 0, 0)
    a = int(total * ratios[0])
    b = int(total * ratios[1])
    c = total - a - b
    return (a, b, c)


def build_input_ids(
    prompt_ids: list[int],
    response_a_ids: list[int],
    response_b_ids: list[int],
    max_len: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    head_ratio: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:
    """回傳截斷後的 (prompt_ids, response_a_ids, response_b_ids)。

    呼叫端還要自己補上 CLS/SEP 特殊 token，所以保證
    len(p) + len(a) + len(b) + NUM_SPECIAL_TOKENS <= max_len。
    """
    budget = max_len - NUM_SPECIAL_TOKENS
    if budget <= 0:
        raise ValueError(f"max_len ({max_len}) 太小，容不下 {NUM_SPECIAL_TOKENS} 個特殊 token")
    p_budget, a_budget, b_budget = split_budget(budget, ratios)
    return (
        truncate_ids(prompt_ids, p_budget, head_ratio),
        truncate_ids(response_a_ids, a_budget, head_ratio),
        truncate_ids(response_b_ids, b_budget, head_ratio),
    )
''',
    'train.py': r'''"""DeBERTa-v3-base 三分類微調：資料集組裝、訓練、推論。

只能在有 torch / transformers 的環境 import（Kaggle Notebook 或有 GPU 的機器）；
本機沒有 GPU，這個檔案沒有、也無法有本機測試覆蓋。截斷邏輯本身在 llmcls/text.py
裡用純 Python 測試過，這裡只是把它接上真正的 tokenizer 和 HF Trainer。

`train_fold()` 是核心入口，同時給兩種呼叫方式用：
- CLI：scripts/train.py（適合遠端 GPU，例如 RunPod）
- Kaggle Notebook cell：`from llmcls.train import train_fold` 直接呼叫
"""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from llmcls.config import MAX_LEN, MODEL_DIR, MODEL_NAME, N_CLASSES, N_FOLDS, SEED
from llmcls.cv import add_folds, fold_indices
from llmcls.data import load_test, load_train
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss
from llmcls.submission import build_submission, save_submission
from llmcls.text import build_input_ids


class PreferenceDataset(torch.utils.data.Dataset):
    """把三欄文字 tokenize 成單一序列：[CLS] prompt [SEP] response_a [SEP] response_b [SEP]。"""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, labels: np.ndarray | None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        # 預先 encode 三欄一次，__getitem__ 只做截斷 + 拼接，不用每個 epoch 重複 tokenize。
        self.prompt_ids = [tokenizer.encode(t, add_special_tokens=False) for t in df["prompt_text"]]
        self.response_a_ids = [tokenizer.encode(t, add_special_tokens=False) for t in df["response_a_text"]]
        self.response_b_ids = [tokenizer.encode(t, add_special_tokens=False) for t in df["response_b_text"]]

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def __getitem__(self, idx: int) -> dict:
        p, a, b = build_input_ids(
            self.prompt_ids[idx], self.response_a_ids[idx], self.response_b_ids[idx], self.max_len
        )
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        input_ids = [cls_id, *p, sep_id, *a, sep_id, *b, sep_id]
        item = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


class StopOnNonFiniteLoss(TrainerCallback):
    """DeBERTa-v3 在 fp32、peak LR 附近實測會突然發散：loss 衝高、grad_norm 變 NaN，
    之後每一步都是壞的，權重永久壞掉但 Trainer 完全不知道、還是把剩下的 epoch 跑完
    （實測浪費了 83 分鐘 GPU 時間裡的 70 分鐘）。這裡一偵測到就叫它停，把剩下的時間
    省下來，`triggered` 讓呼叫端知道這次訓練發散過、權重不可信。
    """

    def __init__(self) -> None:
        self.triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and any(
            isinstance(v, (int, float)) and not math.isfinite(v) for k, v in logs.items() if k in ("loss", "grad_norm")
        ):
            control.should_training_stop = True
            self.triggered = True
        return control


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# log_loss 不可能自然達到的高值，只用來確保「發散」在 metric_for_best_model 排序上
# 一定輸給任何健康的 checkpoint（見 _compute_metrics 的說明）。
_NONFINITE_SENTINEL = 99.0


def _compute_metrics(eval_pred) -> dict:
    """log_loss() 對非法機率值嚴格報錯（這是它的正確行為，見 metrics.py）；但這裡是
    Trainer 的 eval callback，一炸整個 run 就白跑、連權重都存不到，所以不能用 raise。

    早期版本把非有限值夾到均勻機率再算分——結果發散的 checkpoint 剛好算出
    log loss = ln(3)，而 greater_is_better=False 之下 ln(3) 比任何健康 checkpoint
    的分數都「小」，load_best_model_at_end 反而會選中發散的那個。改成回報一個大到
    不可能自然出現的哨兵值，讓發散的 checkpoint 在排序上必輸。
    """
    logits, labels = eval_pred
    probs = softmax(np.asarray(logits))
    finite = np.isfinite(probs).all(axis=1)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        return {"log_loss": _NONFINITE_SENTINEL, "vs_uniform": float("nan"), "n_nonfinite": n_nonfinite}
    score = log_loss(labels, probs)
    return {"log_loss": score, "vs_uniform": UNIFORM_LOGLOSS - score, "n_nonfinite": n_nonfinite}


def _training_args(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_scheduler_type: str = "linear",
    max_steps: int | None = None,
    eval_steps: int | None = None,
) -> TrainingArguments:
    # transformers 把 evaluation_strategy 改名成 eval_strategy 過；Kaggle Notebook 內建的
    # 版本不固定，用 inspect 挑對的參數名比硬編一個更穩。
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    # 一律用 steps（不是 epoch）當 eval/save 的節奏，完整訓練也一樣 —— 實測 DeBERTa-v3
    # 在 fp32 訓練到一半會發散，只在 epoch 邊界存檔的話，發散前那個還健康的檢查點根本
    # 沒機會被存下來，load_best_model_at_end 也就沒有東西可挑。
    default_eval_steps = max(1, max_steps // 2) if max_steps is not None else 500
    steps = eval_steps or default_eval_steps
    kwargs = dict(
        output_dir=str(output_dir),
        **{strategy_key: "steps"},
        save_strategy="steps",
        eval_steps=steps,
        save_steps=steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        # fp16 混合精度在 Kaggle 目前的 transformers/accelerate 組合下，DeBERTa-v3
        # 第一次 backward 就炸「Attempting to unscale FP16 gradients」（已知相容性
        # 問題）。T4 記憶體對 deberta-v3-base + batch_size 8 + max_len 512 綽綽有餘，
        # 先用 fp32 求正確跑通，混合精度是效能優化、不是里程碑 2 的目標。
        fp16=False,
        report_to=[],
        logging_steps=10 if max_steps else 50,
        # 關掉 tqdm 進度條、強制用純文字 print 記錄 loss —— Kaggle Notebook 預設會用
        # rich/widget 進度條，那些輸出只進 __notebook__.ipynb 的 cell output，不會出現
        # 在 `kaggle kernels output` 抓得到的純文字 log，事後完全查不到 loss 曲線。
        disable_tqdm=True,
        seed=42,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return TrainingArguments(**kwargs)


def predict_probs(trainer: Trainer, tokenizer, df: pd.DataFrame, max_len: int) -> np.ndarray:
    """對沒有 label 的 DataFrame（例如 test set）跑推論，回傳 (n, N_CLASSES) 機率。"""
    ds = PreferenceDataset(df, tokenizer, max_len, labels=None)
    logits = trainer.predict(ds).predictions
    return softmax(np.asarray(logits))


def train_fold(
    fold: int = 0,
    model_name: str = MODEL_NAME,
    n_folds: int = N_FOLDS,
    max_len: int = MAX_LEN,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    lr_scheduler_type: str = "linear",
    output_dir: Path | None = None,
    max_train_rows: int | None = None,
    max_valid_rows: int | None = None,
    max_steps: int | None = None,
    eval_steps: int | None = None,
) -> dict:
    """練一個 fold，存權重，回傳 {"score", "n_nonfinite", "diverged", "output_dir",
    "trainer", "tokenizer"}。

    預設只練 fold 0，不是全部 n_folds —— 先確認贏過 baseline_prior.py 印出的分數，
    再決定要不要花時間跑滿整個 CV。

    `max_train_rows` / `max_valid_rows` / `max_steps` / `eval_steps` 是煙霧測試用的：
    隨機抽一小撮資料、跑幾十步就評估一次，把「資料→tokenize→forward→eval→存檔」整條
    路徑在幾分鐘內走過一遍，而不是每次改動都要賭一整個 epoch（30-60 分鐘 GPU 時間）
    才知道炸不炸。跑煙霧測試時務必把 `lr_scheduler_type` 設成 "constant_with_warmup"
    ——用預設的 "linear" 配上 `max_steps` 很小的話，學習率暖身完就立刻開始衰減，
    根本沒有停留在 peak LR 的時間，測不出「訓練到 peak LR 附近才發散」這種問題
    （這正是本專案第一次煙霧測試沒抓到、完整訓練卻在 peak LR 附近整個發散的原因）。
    """
    # Kaggle 的 GPU kernel 預設給 T4 x2；HF Trainer 偵測到多張卡會自動包成
    # nn.DataParallel，這是已知會在 eval 階段的 predictions gather 上出怪問題的來源
    # （一個訊號：「gather along dimension 0 ... all input tensors were scalars」的
    # warning）。deberta-v3-base 在 batch_size 8 / max_len 512 下單張 T4 就跑得動，
    # 沒有 DP 帶來的好處，直接限制成單卡排除這個變因。用 setdefault 而不是強制覆蓋，
    # 呼叫端仍可自行指定 CUDA_VISIBLE_DEVICES。
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    output_dir = Path(output_dir) if output_dir is not None else MODEL_DIR / f"fold{fold}"

    train = load_train()
    print(f"train: {len(train)} 列")
    train = add_folds(train, n_folds=n_folds)
    tr_idx, va_idx = fold_indices(train, fold)
    print(f"fold {fold}: train {len(tr_idx)} 列, valid {len(va_idx)} 列")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=N_CLASSES)
    # 實測 deberta-v3-base 在 HF Hub 上是用 fp16 存的，新版 transformers 的
    # from_pretrained 預設照抄 checkpoint 原本的 dtype，跟 TrainingArguments(fp16=False)
    # 完全無關 —— 結果是在跑「沒有 loss scaler 保護的裸 fp16 訓練」，梯度撐不了多久
    # 就溢位成 NaN，調低學習率只是延後發生、不是解法。強制轉 fp32 才是真正對應
    # fp16=False 的意圖。
    print(f"model 載入時的 dtype：{next(model.parameters()).dtype}")
    model = model.float()

    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    # 用隨機抽樣而不是頭幾列 —— 頭幾列在煙霧測試時永遠是同一批，測不到資料的多樣性。
    if max_train_rows is not None:
        tr_df = tr_df.sample(n=min(max_train_rows, len(tr_df)), random_state=SEED).reset_index(drop=True)
    if max_valid_rows is not None:
        va_df = va_df.sample(n=min(max_valid_rows, len(va_df)), random_state=SEED).reset_index(drop=True)
    tr_ds = PreferenceDataset(tr_df, tokenizer, max_len, tr_df["label"].to_numpy())
    va_ds = PreferenceDataset(va_df, tokenizer, max_len, va_df["label"].to_numpy())

    stop_callback = StopOnNonFiniteLoss()
    trainer = Trainer(
        model=model,
        args=_training_args(
            output_dir, epochs, batch_size, lr, lr_scheduler_type=lr_scheduler_type,
            max_steps=max_steps, eval_steps=eval_steps,
        ),
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[stop_callback],
    )
    trainer.train()
    if stop_callback.triggered:
        print("偵測到 loss/grad_norm 變成 NaN，已提前停止訓練 —— 這次的權重不可信，不要拿去推論")

    # 存檔緊接在 train() 後面、explicit evaluate() 之前 —— load_best_model_at_end=True
    # 已經把最佳權重換回 trainer.model，這裡先存起來，後面的 evaluate() 就算出狀況
    # 也不會白跑一整個 epoch 的 GPU 時間卻什麼都沒留下。
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"模型已存到 {output_dir}")

    metrics = trainer.evaluate()
    score = metrics["eval_log_loss"]
    n_nonfinite = metrics.get("eval_n_nonfinite", 0)
    delta = UNIFORM_LOGLOSS - score
    # n_nonfinite > 0 代表分數是拿均勻機率湊出來的假象（例如全部 clamp 之後 delta 剛好
    # 等於 0，會被誤判成「打平基準」）——只要有非有限值，不管 delta 多少一律算沒過關。
    passed = n_nonfinite == 0 and not stop_callback.triggered and delta > 0
    print(f"\nvalid log loss   {score:.5f}")
    print(f"均勻亂猜基準      {UNIFORM_LOGLOSS:.5f}  (ln 3)")
    print(f"改善              {delta:+.5f}  {'✓ 優於基準' if passed else '✗ 未優於基準'}")
    if n_nonfinite:
        print(f"警告：{n_nonfinite}/{len(va_df)} 筆驗證預測是 NaN/inf，已夾到均勻機率計分 —— 分數不可信，先查訓練穩定性")

    return {
        "score": score,
        "n_nonfinite": n_nonfinite,
        "diverged": stop_callback.triggered,
        "output_dir": output_dir,
        "trainer": trainer,
        "tokenizer": tokenizer,
    }


def load_trained(output_dir: Path):
    """從已存的權重目錄載入 model + tokenizer（供推論 notebook 用，不需要 Trainer）。"""
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir)).float()
    return model, tokenizer


def predict_with_model(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int = 32) -> np.ndarray:
    """離線推論 notebook 用：不需要 Trainer / TrainingArguments，直接跑 forward。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    ds = PreferenceDataset(df, tokenizer, max_len, labels=None)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            inputs = collator(batch).to(device)
            logits = model(**inputs).logits.detach().cpu().numpy()
            all_probs.append(softmax(logits))
    return np.concatenate(all_probs, axis=0)


def predict_test_and_save(trainer_result: dict, name: str = "submission.csv") -> Path:
    """train_fold() 回傳值直接餵進來，對 test.csv 推論並寫出提交檔。"""
    test = load_test()
    probs = predict_probs(trainer_result["trainer"], trainer_result["tokenizer"], test, MAX_LEN)
    sub = build_submission(test["id"], probs)
    return save_submission(sub, name=name)
''',
}

_pkg_dir = pathlib.Path("/kaggle/working/llmcls_src/src/llmcls")
_pkg_dir.mkdir(parents=True, exist_ok=True)
for _name, _content in _llmcls_files.items():
    (_pkg_dir / _name).write_text(_content, encoding="utf-8")

sys.path.insert(0, "/kaggle/working/llmcls_src/src")
print("llmcls bootstrapped:", sorted(p.name for p in _pkg_dir.glob("*.py")))
