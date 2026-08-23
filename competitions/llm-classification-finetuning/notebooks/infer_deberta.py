# %% [markdown]
# # 推論（提交用）—— 5-fold 機率平均 ensemble
#
# **里程碑 3 最後一步**：`notebooks/calibrate_folds.py` 對 5 個 fold 各自量測過
# a/b 對調 TTA + temperature scaling（每個 fold 用自己的驗證集重新配 T），
# 5/5 個 fold 都有改善，平均改善 +0.00836（標準差 0.00288，明顯大於雜訊）：
#
# | fold | baseline | TTA+temperature | T（各自配的） |
# |---|---|---|---|
# | 0 | 1.06839 | 1.05904 | 1.192 |
# | 1 | 1.05459 | 1.04256 | 1.055 |
# | 2 | 1.07392 | 1.07019 | 1.331 |
# | 3 | 1.08028 | 1.07358 | 1.514 |
# | 4 | 1.05712 | 1.04711 | 1.097 |
#
# 上一版（`git log` 可查）只用 fold 0 + TTA + 跨 fold 平均溫度 T=1.238，已送出排行榜
# public score 1.05159（milestone 2 單一 fold、無 TTA/校準是 1.07748）。這一版換成
# **5 個 fold 一起**：每個 fold 各自做 a/b 對調 TTA、各自套自己配出來的溫度（不是全部
# 套同一個跨 fold 平均值——每個溫度是為那個 fold 自己的模型校準的），5 組校準後的
# 機率再取平均，當最終提交的機率。
#
# 這是標準的 bagging 概念：5 個模型看過不同的 80% 訓練子集、隨機初始化跟訓練過程
# 的雜訊也不一樣，會在不完全相同的地方犯錯，機率平均起來能讓誤差互相抵消一部分。
# 但這件事本身**沒辦法在本機乾淨驗證**——每個 fold 的驗證集，另外 4 個模型訓練時
# 都看過，拿它們一起評分會偏樂觀，所以「5 個 fold 平均是否真的比單一 fold 好」只能
# 送一次排行榜才知道。
#
# 推論時間會變成單一 fold 版本的 5 倍（原本幾分鐘 → 預期十幾分鐘），目前判斷還不到
# 需要額外加速的程度。
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
# <slug>/` 而不是網頁 UI 那種 `/kaggle/input/<slug>/`；`dataset_sources` 實測也一樣
# 不是原本猜的 `/kaggle/input/<dataset-slug>/`。下面直接用 glob 找 checkpoint 實際
# 在哪，不管 Kaggle 這次又把它掛在哪個路徑下都能動。
#
# 正式權重掛的是 `fold0-4-checkpoints`（獨立 Kaggle Dataset，不受任何 kernel 之後
# push 影響——見 CLAUDE.md「正式權重只信 Dataset」），檔名是攤平的 `fold{N}__檔名`
# （不是巢狀 `fold{N}/檔名`），下面先把它們還原成 HF `from_pretrained()` 認得的
# 巢狀資料夾（複製到 `/kaggle/working/model/fold{N}/`），再用同一套 glob 邏輯抓。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %%
import pathlib
import shutil

import torch

from llmcls.calibration import apply_temperature
from llmcls.config import MAX_LEN, MODEL_DIR, N_FOLDS
from llmcls.data import load_test
from llmcls.submission import build_submission, save_submission
from llmcls.train import load_trained, predict_logits_with_tta

# 5-fold 各自配出來的溫度（notebooks/calibrate_folds.py 的量測結果，見上面的表格）
# —— 每個溫度是為那個 fold 自己的模型校準的，不是全部套同一個常數。
FOLD_TEMPERATURES = {0: 1.192, 1: 1.055, 2: 1.331, 3: 1.514, 4: 1.097}

for _p in pathlib.Path("/kaggle/input").glob("**/fold*__model.safetensors"):
    fold_name, _ = _p.name.split("__", 1)
    dst = MODEL_DIR / fold_name
    if not dst.exists():
        dst.mkdir(parents=True)
        for _f in _p.parent.glob(f"{fold_name}__*"):
            _, filename = _f.name.split("__", 1)
            shutil.copy(_f, dst / filename)

_fold_checkpoints = {}
for p in sorted(pathlib.Path("/kaggle/input").glob("**/fold*/model.safetensors")):
    fold_num = int(p.parent.name.replace("fold", ""))
    _fold_checkpoints[fold_num] = p.parent
for p in sorted(MODEL_DIR.glob("fold*/model.safetensors")):
    fold_num = int(p.parent.name.replace("fold", ""))
    _fold_checkpoints.setdefault(fold_num, p.parent)

print("找到的 fold checkpoint：")
for fold_num in sorted(_fold_checkpoints):
    print(f"  fold {fold_num}: {_fold_checkpoints[fold_num]}")

_missing = set(range(N_FOLDS)) - set(_fold_checkpoints)
if _missing:
    print("/kaggle/input 底下的項目：", sorted(str(p) for p in pathlib.Path("/kaggle/input").iterdir()))
    raise FileNotFoundError(f"缺少 fold {sorted(_missing)} 的 checkpoint —— 檢查 dataset_sources 是否接對權重 Dataset")

# %%
test = load_test()
print(f"test: {len(test)} 列")  # 正式評分時這裡不會是 3

# 每個 fold：a/b 對調 TTA（兩種順序各推論一次、換回欄位對齊後在 logits 層級平均）
# 接自己的溫度，softmax 之後得到這個 fold 的機率。5 個 fold 的機率最後取平均。
probs_per_fold = []
for fold in sorted(_fold_checkpoints):
    model, tokenizer = load_trained(_fold_checkpoints[fold])
    logits = predict_logits_with_tta(model, tokenizer, test, max_len=MAX_LEN, batch_size=32)
    probs = apply_temperature(logits, FOLD_TEMPERATURES[fold])
    probs_per_fold.append(probs)
    print(f"fold {fold} 推論完成（T={FOLD_TEMPERATURES[fold]}）")

    # 用完這個 fold 的模型就釋放 GPU 記憶體，避免 5 個 fold 依序跑、常駐記憶體疊加。
    del model
    torch.cuda.empty_cache()

ensemble_probs = sum(probs_per_fold) / len(probs_per_fold)

# %%
sub = build_submission(test["id"], ensemble_probs)
path = save_submission(sub)  # 驗證格式 + 寫到 /kaggle/working/submission.csv
print(f"提交檔已寫入並通過格式檢查：{path}  ({len(sub)} 列)")
