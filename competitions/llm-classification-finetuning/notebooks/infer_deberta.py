# %% [markdown]
# # 推論（提交用）
#
# 這是**推論** notebook，跟訓練 notebook（`train_deberta.py`）分開。這題是 code
# competition，正式評分時 Kaggle 會把這個 notebook 的網路關掉，直接重跑一次，
# 並把 `test.csv` 換成真正的隱藏測試集（本機看到的 `test.csv` 只有 3 列，
# 是格式範例，不是真正的評分資料）。所以：
#
# - **不能**從 HuggingFace Hub 下載任何東西，模型必須是掛進來的 Kaggle Dataset。
# - **不能**假設 test.csv 的列數，程式要能吃任意大小的測試集。
# - 只需要 CPU 也能跑就跑得動的 batch size；如果掛了 GPU 就順便用。
#
# Settings：
# - Internet：**Off**
# - Add Data：
#   1. 這個競賽的資料集 `llm-classification-finetuning`（提供真正的 test.csv）
#   2. 訓練 notebook 存出的模型 Dataset（訓練 notebook Save Version 後，在它的
#      Output 分頁能直接 "Add" 成這個 notebook 的輸入）
#
# 下面第一個 code cell 要換成 `notebooks/_bootstrap_cell.py` 的完整內容（跟訓練
# notebook 用同一份；只寫本機檔案、不連網路，離線也能跑）。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %%
import os

# 訓練 notebook 存出的模型 Dataset 掛載路徑；依實際掛載名稱調整。
os.environ["LLMCLS_MODEL_DIR"] = "/kaggle/input/llmcls-deberta-fold0/fold0"

# %%
from llmcls.config import MAX_LEN, MODEL_DIR
from llmcls.data import load_test
from llmcls.submission import build_submission, save_submission
from llmcls.train import load_trained, predict_with_model

model, tokenizer = load_trained(MODEL_DIR)

# %%
test = load_test()
print(f"test: {len(test)} 列")  # 正式評分時這裡不會是 3

probs = predict_with_model(model, tokenizer, test, max_len=MAX_LEN, batch_size=32)

# %%
sub = build_submission(test["id"], probs)
path = save_submission(sub)  # 驗證格式 + 寫到 /kaggle/working/submission.csv
print(f"提交檔已寫入並通過格式檢查：{path}  ({len(sub)} 列)")
