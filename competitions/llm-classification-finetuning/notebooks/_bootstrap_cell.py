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

# Kaggle Notebook 內資料會掛在 /kaggle/input/<competition>/；本機則放在 ./data/。
# 可用環境變數 LLMCLS_DATA_DIR 覆寫（例如指到 data/fixture 跑煙霧測試）。
_KAGGLE_INPUT = Path("/kaggle/input") / COMPETITION
# 本競賽目錄 competitions/<slug>/，不是 git repo 根目錄 —— 工作區還有其他競賽。
_COMP_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_dir() -> Path:
    if env := os.environ.get("LLMCLS_DATA_DIR"):
        return Path(env)
    if _KAGGLE_INPUT.exists():
        return _KAGGLE_INPUT
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from llmcls.config import MAX_LEN, MODEL_DIR, MODEL_NAME, N_CLASSES, N_FOLDS
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


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = softmax(np.asarray(logits))
    score = log_loss(labels, probs)
    return {"log_loss": score, "vs_uniform": UNIFORM_LOGLOSS - score}


def _training_args(output_dir: Path, epochs: int, batch_size: int, lr: float) -> TrainingArguments:
    # transformers 把 evaluation_strategy 改名成 eval_strategy 過；Kaggle Notebook 內建的
    # 版本不固定，用 inspect 挑對的參數名比硬編一個更穩。
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    return TrainingArguments(
        output_dir=str(output_dir),
        **{strategy_key: "epoch"},
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to=[],
        logging_steps=50,
        seed=42,
    )


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
    output_dir: Path | None = None,
) -> dict:
    """練一個 fold，存權重，回傳 {"score", "output_dir", "trainer", "tokenizer"}。

    預設只練 fold 0，不是全部 n_folds —— 先確認贏過 baseline_prior.py 印出的分數，
    再決定要不要花時間跑滿整個 CV。
    """
    output_dir = Path(output_dir) if output_dir is not None else MODEL_DIR / f"fold{fold}"

    train = load_train()
    print(f"train: {len(train)} 列")
    train = add_folds(train, n_folds=n_folds)
    tr_idx, va_idx = fold_indices(train, fold)
    print(f"fold {fold}: train {len(tr_idx)} 列, valid {len(va_idx)} 列")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=N_CLASSES)

    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    tr_ds = PreferenceDataset(tr_df, tokenizer, max_len, tr_df["label"].to_numpy())
    va_ds = PreferenceDataset(va_df, tokenizer, max_len, va_df["label"].to_numpy())

    trainer = Trainer(
        model=model,
        args=_training_args(output_dir, epochs, batch_size, lr),
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    score = metrics["eval_log_loss"]
    delta = UNIFORM_LOGLOSS - score
    print(f"\nvalid log loss   {score:.5f}")
    print(f"均勻亂猜基準      {UNIFORM_LOGLOSS:.5f}  (ln 3)")
    print(f"改善              {delta:+.5f}  {'✓ 優於基準' if delta > 0 else '✗ 未優於基準'}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"模型已存到 {output_dir}")

    return {"score": score, "output_dir": output_dir, "trainer": trainer, "tokenizer": tokenizer}


def load_trained(output_dir: Path):
    """從已存的權重目錄載入 model + tokenizer（供推論 notebook 用，不需要 Trainer）。"""
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))
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
