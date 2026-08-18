# %% [markdown]
# # 訓練 DeBERTa-v3-base（里程碑 2）
#
# 這是**訓練** notebook，跟推論 notebook（`infer_deberta.py`）分開 —— 訓練需要連網路
# 從 HuggingFace Hub 下載 base model 權重，但這題是 code competition，正式推論時
# Kaggle 會**關閉網路**，所以離線推論必須另開一個 notebook，把這裡存出的權重當
# Kaggle Dataset 掛進去用。
#
# Settings：
# - Accelerator：GPU T4 x2（或 P100）
# - Internet：On
# - Add Data：這個競賽的資料集 `llm-classification-finetuning`
#
# 把 repo 的 `src/llmcls/`、`scripts/train.py` 上傳成一個 Kaggle Dataset（例如取名
# `llmcls-src`）掛進來，或者最省事：把 `src/llmcls/*.py` 的內容直接貼進下面標
# 「貼上 llmcls 原始碼」的 cell。

# %%
import sys

# 依實際掛載路徑調整；若用「直接貼原始碼」的做法就不需要這行。
sys.path.insert(0, "/kaggle/input/llmcls-src/src")

# %% [markdown]
# 若 Kaggle 內建的 transformers 版本太舊，才需要下面這行（通常不必要，內建已經夠新）。

# %%
# !pip install -q -U transformers accelerate

# %%
from llmcls.train import train_fold

# 先只練 fold 0。valid log loss 必須小於 1.09861（ln 3）—— 這是本專案判斷分數的
# 唯一標準，也是 scripts/baseline_prior.py 在真實資料上印出的基準（1.09723）。
# 如果沒贏過，先別急著跑其他 fold，回頭檢查 max_len / 學習率。
result = train_fold(fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5)

# %% [markdown]
# `train_fold()` 已經把權重存到 `MODEL_DIR/fold0`（預設 `/kaggle/working/model/fold0`），
# 印出的 valid log loss 也已經跟 ln(3) 比較過。
#
# 確認贏過基準之後：**Save Version**（Save & Run All），存出的 Version 的
# `/kaggle/working/model/` 就會變成一個新的 Kaggle Dataset（在 Notebook 的 Output
# 分頁），下一步把它掛進 `infer_deberta.py`。
#
# 之後要跑滿 5 folds，就是把上面的 `fold=0` 換成 0..4 各跑一次，每個 fold 都會分別
# 存到 `model/fold{N}/`；推論時對 5 個 fold 的機率取平均（等 milestone 3 再做，這裡
# 先求有一個 fold 能打敗基準）。

# %%
print(f"fold 0 valid log loss: {result['score']:.5f}")
