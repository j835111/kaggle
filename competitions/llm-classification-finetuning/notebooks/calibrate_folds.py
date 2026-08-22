# %% [markdown]
# # 里程碑 3：TTA（a/b 對調）與 temperature scaling 校準實驗 —— 5 folds
#
# 這是**實驗量測**用的 notebook，不重新訓練，只對已經存好的 5 個 fold 權重、各自在
# 自己的驗證集（`fold_indices(fold=N)` 切出來的 held-out 那份，訓練時從沒看過）上
# 量測四種組合的 log loss：
#
# 1. baseline —— 原始順序，沒有 TTA、沒有校準
# 2. TTA —— response_a/response_b 對調各推論一次，換回欄位對齊後在機率層級平均
# 3. temperature scaling —— **每個 fold 各自用自己的驗證集配溫度 T**（不是套用其他
#    fold 配出來的常數——每個 fold 的驗證集對那個 fold 的模型來說才是 held-out，
#    各自獨立配 T 才公平，套用別的 fold 配出來的 T 等於在「不是那個 fold 的
#    held-out 資料」上驗證，會失去 held-out 的意義）
# 4. TTA + temperature —— 先把兩個順序的 logits 在 logits 層級平均，再對合併後的
#    logits 配溫度
#
# 只在 fold 0 量過一次（1.08495 → 1.08112，改善 +0.00383）不夠：那份驗證集有可能
# 剛好對這個手法有利，5 個 fold 各自獨立量測同一個改善才有說服力。**但即使 5 個
# fold 都一致改善，也只代表這不是單一切分的巧合，不代表效果量必然完全轉移到隱藏
# 測試集**——5 個 fold 共用同一套模型/訓練 recipe/資料分布，不是 5 個真正獨立的
# 實驗，這點在下面的結論裡會用比較保守的說法。
#
# 用 `kaggle kernels push` 上傳（見 notebooks/README.md 的模式）：`competition_sources`
# 掛這場競賽的資料（重建每個 fold 的驗證切分要用到 train.csv），`kernel_sources` 接
# 練滿 5 folds 的訓練 kernel 輸出，**關網路**（讀本地權重，不用連 HuggingFace Hub）。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %% [markdown]
# **不要猜掛載路徑**——跟 `infer_deberta.py` 一樣的坑，直接 glob 找全部 fold 的
# checkpoint 實際在哪。

# %%
import pathlib

_fold_checkpoints = {}
for p in sorted(pathlib.Path("/kaggle/input").glob("**/fold*/model.safetensors")):
    fold_num = int(p.parent.name.replace("fold", ""))
    _fold_checkpoints[fold_num] = p.parent

print("找到的 fold checkpoint：")
for fold_num in sorted(_fold_checkpoints):
    print(f"  fold {fold_num}: {_fold_checkpoints[fold_num]}")
if not _fold_checkpoints:
    print("/kaggle/input 底下的項目：", sorted(str(p) for p in pathlib.Path("/kaggle/input").iterdir()))
    raise FileNotFoundError("找不到任何 fold*/model.safetensors —— 檢查 kernel_sources 是否正確接上訓練 kernel")

# %% [markdown]
# 對每個 fold：重建它自己的驗證切分（`add_folds` / `fold_indices` 是同一套決定性
# 切分邏輯，重跑一次會得到一模一樣的切分）、載入它自己的 checkpoint、量測四種組合，
# T 用這個 fold 自己的驗證集重新配。兩次推論（原始順序、對調順序）都只做一次，
# 存下原始 logits，後面的組合全部從這兩組 logits 算出來，不重複跑 forward。

# %%
import torch

from llmcls.calibration import apply_temperature, fit_temperature
from llmcls.config import MAX_LEN, N_FOLDS
from llmcls.cv import add_folds, fold_indices
from llmcls.data import load_train
from llmcls.metrics import log_loss, softmax
from llmcls.train import load_trained, predict_logits_with_model, swap_ab
from llmcls.tta import average_swapped

train = load_train()
train = add_folds(train, n_folds=N_FOLDS)

per_fold = {}
for fold in sorted(_fold_checkpoints):
    _, va_idx = fold_indices(train, fold=fold)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    labels = va_df["label"].to_numpy()

    model, tokenizer = load_trained(_fold_checkpoints[fold])

    logits_orig = predict_logits_with_model(model, tokenizer, va_df, max_len=MAX_LEN, batch_size=32)
    logits_swapped = predict_logits_with_model(model, tokenizer, swap_ab(va_df), max_len=MAX_LEN, batch_size=32)

    # 用完這個 fold 的模型就釋放 GPU 記憶體，避免 5 個 fold 依序跑、常駐記憶體疊加。
    del model
    torch.cuda.empty_cache()

    probs_orig = softmax(logits_orig)
    probs_swapped_aligned = softmax(logits_swapped)[:, [1, 0, 2]]

    baseline_loss = log_loss(labels, probs_orig)
    swapped_only_loss = log_loss(labels, probs_swapped_aligned)
    tta_loss = log_loss(labels, average_swapped(probs_orig, softmax(logits_swapped)))

    t_baseline = fit_temperature(logits_orig, labels)
    temp_loss = log_loss(labels, apply_temperature(logits_orig, t_baseline))

    combined_logits = average_swapped(logits_orig, logits_swapped)
    t_combined = fit_temperature(combined_logits, labels)
    tta_temp_loss = log_loss(labels, apply_temperature(combined_logits, t_combined))

    per_fold[fold] = {
        "n_valid": len(va_df),
        "baseline": baseline_loss,
        "swapped_only": swapped_only_loss,
        "tta": tta_loss,
        "temperature": temp_loss,
        "t_baseline": t_baseline,
        "tta_temperature": tta_temp_loss,
        "t_combined": t_combined,
    }
    print(
        f"fold {fold}（{len(va_df)} 列驗證集）：baseline {baseline_loss:.5f}  "
        f"TTA {tta_loss:.5f}  temp(T={t_baseline:.3f}) {temp_loss:.5f}  "
        f"TTA+temp(T={t_combined:.3f}) {tta_temp_loss:.5f}"
    )

# %% [markdown]
# 彙總 5 個 fold：每個組合的平均/標準差、T 值跨 fold 穩不穩定、TTA+temperature
# 相對 baseline 的改善是不是每個 fold 都成立。

# %%
import numpy as np

metrics_keys = ["baseline", "swapped_only", "tta", "temperature", "tta_temperature"]
print("=== 5 folds 明細 ===")
header = f"{'fold':<6}" + "".join(f"{k:>16}" for k in metrics_keys)
print(header)
for fold in sorted(per_fold):
    r = per_fold[fold]
    print(f"{fold:<6}" + "".join(f"{r[k]:>16.5f}" for k in metrics_keys))

print("\n=== 跨 fold 平均 / 標準差 ===")
for k in metrics_keys:
    vals = np.array([per_fold[f][k] for f in per_fold])
    print(f"{k:<16}  mean={vals.mean():.5f}  std={vals.std():.5f}")

t_values = np.array([per_fold[f]["t_combined"] for f in per_fold])
print(f"\nT（TTA+temperature，每個 fold 各自配）：{list(np.round(t_values, 3))}")
print(f"  mean={t_values.mean():.3f}  std={t_values.std():.3f}")
if t_values.std() < 0.15:
    print(f"  T 在各 fold 間穩定，硬編碼 T≈{t_values.mean():.3f} 到 infer_deberta.py 算合理")
else:
    print("  T 在各 fold 間變動較大，硬編碼單一常數不夠穩，建議用跨 fold 平均值")

improvement = np.array([per_fold[f]["baseline"] - per_fold[f]["tta_temperature"] for f in per_fold])
n_improved = int((improvement > 0).sum())
print(f"\nTTA+temperature 相對 baseline 的改善（5 folds）：{list(np.round(improvement, 5))}")
print(f"平均改善 {improvement.mean():+.5f}  標準差 {improvement.std():.5f}  {n_improved}/{len(improvement)} 個 fold 有改善")

print(
    "\n注意：5 個 fold 共用同一套模型/訓練 recipe/資料分布，不是 5 個獨立實驗——"
    "一致改善代表這不是單一切分的巧合，不代表效果量必然完全轉移到隱藏測試集。"
)
