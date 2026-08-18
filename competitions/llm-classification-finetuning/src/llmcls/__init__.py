"""LLM Classification Finetuning — 共用工具模組。

刻意寫成可 import 的模組（而不是單一 notebook），因為訓練會在
Kaggle Notebook 或遠端 GPU 上跑，本機只負責資料處理、CV 切分與提交組裝。
同一份程式碼兩邊都能 import，路徑差異由 config.py 吸收。
"""

from llmcls.config import DATA_DIR, LABEL_COLS, OUTPUT_DIR
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss

__all__ = ["DATA_DIR", "OUTPUT_DIR", "LABEL_COLS", "log_loss", "UNIFORM_LOGLOSS"]
