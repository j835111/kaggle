# %% [markdown]
# # 推論（提交用）
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
