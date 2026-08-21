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
# **先跑煙霧測試，不要直接跑整個 epoch。** 里程碑 2 第一次完整訓練（fold 0、
# batch_size=8、fp32、每 500 步對完整驗證集算一次分數）花了將近 9 小時，其中光是
# 評估就佔掉將近一半 —— 這裡在原本已經驗證過穩定的設定上加兩個加速手段，跑之前先用
# 煙霧測試確認沒有引入新的不穩定：
#
# 1. `fp16=True`：train_fold() 已經會先強制 `model.float()` 轉成真正的 fp32，
#    這裡的 fp16 是正規的 autocast + GradScaler 混合精度 —— 跟里程碑 2 除錯過程中
#    踩到的「裸 fp16、沒有 loss scaler」完全不同，梯度真的非有限值時 GradScaler
#    會跳過那一步而不是把權重弄壞，`StopOnNonFiniteLoss` 仍是最後一道防線。
# 2. `eval_subset_rows`：訓練中途的週期性評估改成只對一小撮驗證集跑（快很多），
#    最後才對完整驗證集重新算一次真正的分數 —— 發散偵測本來就跟評估頻率無關
#    （`StopOnNonFiniteLoss` 每 50 步看一次 loss/grad_norm），拉開評估頻率不會
#    犧牲安全網。
#
# `batch_size` 原本也想從 8 調到 16 —— 煙霧測試（3000 筆子集）完全穩定通過，但換成
# 完整的 45746 筆資料後，訓練中途在某一批剛好全是接近 max_len 上限的長序列時 CUDA
# OOM 了（T4 記憶體只差 66MB）。**煙霧測試只能驗證穩定性，驗不出完整資料集上的
# 記憶體上限** —— 子集抽樣很難剛好抽到最壞情況的那幾批，所以 `batch_size` 維持
# 里程碑 2 已經在完整資料集上證明過安全的 8，不冒這個風險。
#
# 煙霧測試維持跟里程碑 2 一樣的做法：`lr_scheduler_type="constant_with_warmup"`、
# 跑 400 步，讓學習率停在 peak 不再衰減，才測得出「訓練到 peak LR 附近才發散」
# 這種問題。

# %%
from llmcls.train import train_fold

smoke = train_fold(
    fold=0, epochs=1, batch_size=8, max_len=512, lr=2e-5,
    lr_scheduler_type="constant_with_warmup", fp16=True,
    max_train_rows=3000, max_valid_rows=800, max_steps=400, eval_steps=200,
    output_dir="/kaggle/working/model/_smoke",
)
print(f"log loss={smoke['score']:.5f}  n_nonfinite={smoke['n_nonfinite']}  diverged={smoke['diverged']}")
assert not smoke["diverged"] and smoke["n_nonfinite"] == 0, (
    "加速設定（fp16=True）在煙霧測試就不穩定，先不要跑完整訓練"
)
print("煙霧測試通過，加速設定沒有引入不穩定")

# %% [markdown]
# 煙霧測試過關後才跑完整訓練。valid log loss 必須小於 1.09861（ln 3）—— 這是本專案
# 判斷分數的唯一標準，也是 scripts/baseline_prior.py 在真實資料上印出的基準
# （1.09723）。步進式存檔（每 1500 步）跟發散偵測保護都還在，不會再白燒一整個
# epoch。`eval_subset_rows=2000` 讓訓練中途的評估變快，最終回報的分數保證是對
# 完整驗證集算出來的。

# %%
import time

_t0 = time.time()
result = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
)
print(f"訓練總耗時：{time.time() - _t0:.0f}s")
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
