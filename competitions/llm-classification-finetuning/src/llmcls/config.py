"""路徑與常數。本機 / Kaggle Notebook 的差異全部集中在這裡。"""

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
