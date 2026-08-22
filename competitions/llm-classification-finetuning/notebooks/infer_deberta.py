# %% [markdown]
# # 推論（提交用）
#
# **里程碑 3 更新**：`notebooks/calibrate_folds.py` 對 5 個 fold 各自量測過
# a/b 對調 TTA + temperature scaling（每個 fold 用自己的驗證集重新配 T），
# 5/5 個 fold 都有改善，平均改善 +0.00836（標準差 0.00288，明顯大於雜訊）：
#
# | fold | baseline | TTA+temperature | T（各自配的） |
# |---|---|---|---|
# | 0（推論用這個） | 1.06839 | **1.05904** | 1.192 |
# | 1 | 1.05459 | 1.04256 | 1.055 |
# | 2 | 1.07392 | 1.07019 | 1.331 |
# | 3 | 1.08028 | 1.07358 | 1.514 |
# | 4 | 1.05712 | 1.04711 | 1.097 |
#
# T 值跨 fold 差異不小（1.055~1.514，平均 1.238，標準差 0.167）——單一 fold 配出來
# 的 T 對那個 fold 來說最準，但拿掉單一 fold 的雜訊之後，跨 fold 平均是更穩的估計，
# 所以套用時用 **T=1.238**（5-fold 平均），不是 fold 0 自己配出來的 1.192。
#
# fold 0 本身也在這次搶救 5-fold 訓練時被重新訓練過一次（驗證分數從 1.08494
# 降到 1.06839），跟 milestone 2 送出 143/212 那次用的權重不是同一份——這是
# 目前這個推論版本相對上一次排行榜結果的主要改善來源，TTA/校準是疊加上去的
# 第二層改善。（5 個 fold 共用同一套模型/recipe/資料分布，不是 5 個獨立實驗，
# 一致改善代表這不是單一切分的巧合，但不保證效果量完全轉移到隱藏測試集。）
#
# 這是**推論** notebook，跟訓練 notebook（`train_deberta.py`）分開。這題是 code
# competition，正式評分時 Kaggle 會把這個 notebook 的網路關掉，直接重跑一次，
# 並把 `test.csv` 換成真正的隱藏測試集（本機看到的 `test.csv` 只有 3 列，
# 是格式範例，不是真正的評分資料）。所以：
#
# - **不能**從 HuggingFace Hub 下載任何東西，模型必須是掛進來的資料。
# - **不能**假設 test.csv 的列數，程式要能吃任意大小的測試集。
# - 只需要 CPU 也能跑就跑得動的 batch size；如果掛了 GPU 就順便用。
#
# 用 `kaggle kernels push` 上傳時（見 notebooks/README.md），這個 notebook 的
# kernel-metadata.json 已經用 `kernel_sources` 把訓練 kernel的輸出接進來，
# `competition_sources` 也已經接了這個競賽的資料，`test.csv` 會是真正的隱藏測試集。
#
# **不要猜掛載路徑**——`competition_sources` 實測掛在 `/kaggle/input/competitions/
# <slug>/` 而不是網頁 UI 那種 `/kaggle/input/<slug>/`；`kernel_sources` 實測也一樣
# 不是原本猜的 `/kaggle/input/<kernel-slug>/`。下面直接用 glob 找 checkpoint 實際
# 在哪，不管 Kaggle 這次又把它掛在哪個路徑下都能動。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %%
import os
import pathlib

_candidates = sorted(pathlib.Path("/kaggle/input").glob("**/fold0/model.safetensors"))
print("找到的 fold0 checkpoint：", _candidates)
if not _candidates:
    print("/kaggle/input 底下的項目：", sorted(str(p) for p in pathlib.Path("/kaggle/input").iterdir()))
    raise FileNotFoundError("找不到 fold0/model.safetensors —— 檢查 kernel_sources 是否正確接上訓練 kernel")
os.environ["LLMCLS_MODEL_DIR"] = str(_candidates[0].parent)
print("LLMCLS_MODEL_DIR =", os.environ["LLMCLS_MODEL_DIR"])

# %%
from llmcls.calibration import apply_temperature
from llmcls.config import MAX_LEN, MODEL_DIR
from llmcls.data import load_test
from llmcls.submission import build_submission, save_submission
from llmcls.train import load_trained, predict_logits_with_tta

# 5-fold 平均溫度（notebooks/calibrate_folds.py 的量測結果，見上面的說明），
# 不是 fold 0 自己配出來的 1.192——跨 fold 平均比單一 fold 的估計更穩。
FITTED_TEMPERATURE = 1.238

model, tokenizer = load_trained(MODEL_DIR)

# %%
test = load_test()
print(f"test: {len(test)} 列")  # 正式評分時這裡不會是 3

# a/b 對調 TTA：兩種順序各推論一次、換回欄位對齊後在 logits 層級平均，
# 再對合併後的 logits 套用配好的溫度。
logits = predict_logits_with_tta(model, tokenizer, test, max_len=MAX_LEN, batch_size=32)
probs = apply_temperature(logits, FITTED_TEMPERATURE)

# %%
sub = build_submission(test["id"], probs)
path = save_submission(sub)  # 驗證格式 + 寫到 /kaggle/working/submission.csv
print(f"提交檔已寫入並通過格式檢查：{path}  ({len(sub)} 列)")
