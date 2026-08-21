# %% [markdown]
# # 推論（提交用）
#
# **里程碑 3 更新**：套用 `notebooks/calibrate_fold0.py` 在 fold 0 驗證集（11731 列）
# 上量測到的最佳組合——a/b 對調 TTA + temperature scaling（T=1.429）：
#
# | 組合 | valid log loss | 相對 baseline |
# |---|---|---|
# | baseline（milestone 2，已送出 143/212） | 1.08495 | — |
# | 單獨對調順序（診斷位置偏誤用） | 1.08439 | -0.00056 |
# | TTA（a/b 對調平均） | 1.08416 | +0.00078 |
# | temperature scaling | 1.08157 | +0.00338 |
# | **TTA + temperature（目前用的）** | **1.08112** | **+0.00383** |
#
# 位置偏誤本身很小（單獨對調順序只差 0.00056，遠低於 0.01），TTA 單獨效果有限，
# 校準才是主要來源；兩者疊加仍然比只用校準略好，所以兩個一起套用。T 值是在 fold 0
# 自己的 held-out 驗證集上配的（只有一個純量參數，held-out 資料上配它不算作弊），
# 不是在這個 notebook 看得到的 test.csv 上配的。
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

# fold 0 驗證集（11731 列）上配出來的溫度（notebooks/calibrate_fold0.py 的
# TTA + temperature 組合，valid log loss 1.08112，見上面的說明）。
FITTED_TEMPERATURE = 1.429

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
