# %% [markdown]
# # 訓練 DeBERTa-v3-base（里程碑 2）
#
# 這是**訓練** notebook，跟推論 notebook（`infer_deberta.py`）分開 —— 訓練需要連網路
# 從 HuggingFace Hub 下載 base model 權重，但這題是 code competition，正式推論時
# Kaggle 會**關閉網路**，所以離線推論必須另開一個 notebook，把這裡存出的權重當
# Kaggle Dataset 掛進去用。
#
# 用 `kaggle kernels push` 上傳時（見 notebooks/README.md），kernel-metadata.json
# 已經設好 GPU、Internet On、`competition_sources` 掛這個競賽的資料 —— 不需要手動
# 在網頁上調整。手動在 Kaggle 網頁貼 cell 的話才需要自己設：
# - Accelerator：GPU T4 x2（或 P100）
# - Internet：On
# - Add Data：這個競賽的資料集 `llm-classification-finetuning`
#
# 下面第一個 code cell 要換成 `notebooks/_bootstrap_cell.py` 的完整內容（用
# `python scripts/gen_notebook_bootstrap.py > notebooks/_bootstrap_cell.py` 產生，
# `src/llmcls/` 有改動就要重新產生一次再貼）。它會把 `src/llmcls/*.py` 的原始碼直接
# 寫進 `/kaggle/working/llmcls_src/`，不需要另外建 Kaggle Dataset 掛程式碼
# ——掛 Dataset 那條路容易在「有沒有建對 / 掛載名稱對不對」上出錯（`ModuleNotFoundError`）。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %% [markdown]
# 檢查一下 `/kaggle/input` 底下實際掛了什麼、`llmcls.config` 解析出來的路徑對不對 ——
# `competition_sources` 掛的資料如果不在 `DATA_DIR` 猜的路徑，`load_train()` 會找不到
# `train.csv`，這個 cell 就是拿來當場抓出真正的掛載路徑用的。

# %%
import pathlib

print("/kaggle/input 底下（找 train.csv，最多往下 3 層）：")
for p in pathlib.Path("/kaggle/input").glob("**/train.csv"):
    print(" ", p)

from llmcls.config import DATA_DIR

print("llmcls.config.DATA_DIR =", DATA_DIR, " exists =", DATA_DIR.exists())
if DATA_DIR.exists():
    print("DATA_DIR 底下：", sorted(p.name for p in DATA_DIR.iterdir()))

# %% [markdown]
# 若 Kaggle 內建的 transformers 版本太舊，才需要下面這行（通常不必要，內建已經夠新）。

# %%
# !pip install -q -U transformers accelerate

# %% [markdown]
# **先跑煙霧測試，不要直接跑整個 epoch。** 完整訓練一個 epoch 在 T4 上要 30-60
# 分鐘，改一行程式碼就要賭這麼久才知道有沒有炸，太貴 —— 前幾次完整訓練都實測發散
# 過（grad_norm 變 NaN、權重永久壞掉），一路查下來真正的原因是：deberta-v3-base
# 在 HF Hub 上是用 **fp16** 存的，新版 transformers 的 `from_pretrained` 預設照抄
# checkpoint 原本的 dtype，跟 `TrainingArguments(fp16=False)` 完全無關 —— 等於一直
# 在跑「沒有 loss scaler 保護的裸 fp16 訓練」，梯度撐不了多久就溢位成 NaN。
# `train_fold()` 現在會強制 `.float()`，並且印出實際的 dtype 供確認。
#
# 煙霧測試刻意把 `lr_scheduler_type` 設成 `"constant_with_warmup"`、跑 400 步：
# 暖身後學習率會停在 peak LR 不再衰減，才測得出「訓練到 peak LR 附近才發散」這種
# 問題；步數也拉大到 400（實測發散發生在 peak LR 之後 40~260 步不等，100 步太短，
# 有一次沒抓到）。lr 沿用完整訓練的 2e-5；如果 dtype 修好了還是發散，才降到 1e-5。

# %%
from llmcls.train import train_fold


def _smoke(lr: float) -> dict:
    print(f"--- 煙霧測試：lr={lr}, constant_with_warmup, max_steps=400 ---")
    r = train_fold(
        fold=0, epochs=1, batch_size=8, max_len=512, lr=lr,
        lr_scheduler_type="constant_with_warmup",
        max_train_rows=3000, max_valid_rows=800, max_steps=400, eval_steps=200,
        output_dir="/kaggle/working/model/_smoke",
    )
    print(f"lr={lr}: log loss={r['score']:.5f}  n_nonfinite={r['n_nonfinite']}  diverged={r['diverged']}")
    return r


smoke = _smoke(2e-5)
train_lr = 2e-5
if smoke["diverged"] or smoke["n_nonfinite"] > 0:
    print("lr=2e-5 在煙霧測試還是發散，改用 1e-5 重試（dtype 應該已經修好，這是次要防線）")
    smoke = _smoke(1e-5)
    train_lr = 1e-5

assert not smoke["diverged"] and smoke["n_nonfinite"] == 0, (
    f"lr={train_lr} 煙霧測試仍然發散，dtype 修復可能沒生效或另有原因，先不要跑完整訓練"
)
print(f"煙霧測試通過，完整訓練用 lr={train_lr}")

# %% [markdown]
# 煙霧測試過關後才跑完整訓練，用煙霧測試驗證過的 `train_lr`。valid log loss 必須
# 小於 1.09861（ln 3）—— 這是本專案判斷分數的唯一標準，也是 scripts/baseline_prior.py
# 在真實資料上印出的基準（1.09723）。就算煙霧測試過了，完整訓練還是有步進式存檔
# （每 500 步）跟發散偵測保護，不會再像第一次那樣白燒一整個 epoch。

# %%
result = train_fold(fold=0, epochs=2, batch_size=8, max_len=512, lr=train_lr, eval_steps=500)
assert not result["diverged"] and result["n_nonfinite"] == 0, "完整訓練發散或有非有限值，模型權重不可信，不要拿去推論"

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
