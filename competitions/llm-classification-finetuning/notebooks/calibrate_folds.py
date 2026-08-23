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
# 掛這場競賽的資料（重建每個 fold 的驗證切分要用到 train.csv），`dataset_sources` 接
# 涵蓋全部 5 folds 的永久 Dataset（`fold0-4-checkpoints`，見 CLAUDE.md「正式權重只信
# Dataset」），**關網路**（讀本地權重，不用連 HuggingFace Hub）。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %% [markdown]
# **不要猜掛載路徑**——跟 `infer_deberta.py` 一樣的坑，直接 glob 找全部 fold 的
# checkpoint 實際在哪。Dataset 裡的檔名是攤平的 `fold{N}__檔名`（不是巢狀
# `fold{N}/檔名`），先還原成 HF `from_pretrained()` 認得的巢狀資料夾。

# %%
import pathlib
import shutil

from llmcls.config import MODEL_DIR

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
if not _fold_checkpoints:
    print("/kaggle/input 底下的項目：", sorted(str(p) for p in pathlib.Path("/kaggle/input").iterdir()))
    raise FileNotFoundError("找不到任何 fold checkpoint —— 檢查 dataset_sources 是否正確接上權重 Dataset")

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

    probs_tta_temp = apply_temperature(combined_logits, t_combined)

    per_fold[fold] = {
        "n_valid": len(va_df),
        "baseline": baseline_loss,
        "swapped_only": swapped_only_loss,
        "tta": tta_loss,
        "temperature": temp_loss,
        "t_baseline": t_baseline,
        "tta_temperature": tta_temp_loss,
        "t_combined": t_combined,
        # winner_tie 診斷用：留住這個 fold 的原始標籤跟兩種機率，之後要拼起來按
        # 真實類別（0=winner_model_a、1=winner_model_b、2=winner_tie，對應
        # llmcls.config.LABEL_COLS 的 argmax 編碼）分開算 log loss。
        "labels": labels,
        "probs_baseline": probs_orig,
        "probs_tta_temp": probs_tta_temp,
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

# %% [markdown]
# ## winner_tie 診斷：損失主要出在哪一類？
#
# 5-fold 交叉驗證下，每一列訓練資料剛好被當過一次（且僅一次）驗證集——把 5 個 fold
# 的驗證集預測全部拼在一起，等於用**全部** `train.csv` 做了一次公平的 held-out 檢驗
# （每列都是某個 fold 訓練時沒看過的資料）。用這份拼起來的結果，照真實答案分成
# `winner_model_a`（label=0）/`winner_model_b`（label=1）/`winner_tie`（label=2）
# 三組分別算 log loss，才能看出損失是不是集中在某一類——如果 tie 明顯比另外兩類差
# 很多，才值得投入 tie 專屬的處理（例如額外的類別權重、focal loss、或針對 tie 的
# 特徵工程）；如果三類其實差不多，這個方向就不值得做，milestone 3 的力氣該留給
# 別的候選手法。
#
# 同時列出 baseline（沒做 TTA/校準的原始模型）跟 TTA+temperature（`infer_deberta.py`
# 正式上線用的組合）兩種算法的每類別分數——這樣才看得出來 TTA+校準對三個類別是
# 平均改善，還是剛好特別修正了其中一類（例如 TTA 對調 a/b 順序，理論上只會影響
# a/b 兩類的順序偏見，對 tie 這種「兩邊打平」的情況未必有幫助）。

# %%
import numpy as np

from llmcls.config import LABEL_COLS

all_labels = np.concatenate([per_fold[f]["labels"] for f in sorted(per_fold)])
all_probs_baseline = np.concatenate([per_fold[f]["probs_baseline"] for f in sorted(per_fold)], axis=0)
all_probs_tta_temp = np.concatenate([per_fold[f]["probs_tta_temp"] for f in sorted(per_fold)], axis=0)

print(f"拼接後總列數：{len(all_labels)}（應該等於 train.csv 的總列數，5 個 fold 互不重疊、剛好覆蓋全部）")
print(f"整體 baseline log loss（拼接後重算，應該接近 5 個 fold baseline 的加權平均）："
      f"{log_loss(all_labels, all_probs_baseline):.5f}")
print(f"整體 TTA+temperature log loss（拼接後重算）："
      f"{log_loss(all_labels, all_probs_tta_temp):.5f}")

print("\n=== 按真實類別拆解（全部 5 folds 拼接後）===")
print(f"{'類別':<18}{'筆數':>8}{'佔比':>8}{'baseline':>12}{'TTA+temp':>12}{'改善':>10}")
per_class_summary = {}
for class_idx, class_name in enumerate(LABEL_COLS):
    mask = all_labels == class_idx
    n = int(mask.sum())
    loss_baseline = log_loss(all_labels[mask], all_probs_baseline[mask])
    loss_tta_temp = log_loss(all_labels[mask], all_probs_tta_temp[mask])
    per_class_summary[class_name] = {
        "n": n, "baseline": loss_baseline, "tta_temperature": loss_tta_temp,
    }
    print(
        f"{class_name:<18}{n:>8}{n / len(all_labels):>8.1%}"
        f"{loss_baseline:>12.5f}{loss_tta_temp:>12.5f}{loss_baseline - loss_tta_temp:>+10.5f}"
    )

worst_class = max(per_class_summary, key=lambda k: per_class_summary[k]["tta_temperature"])
print(
    f"\nTTA+temperature 之後，log loss 最高的類別是「{worst_class}」"
    f"（{per_class_summary[worst_class]['tta_temperature']:.5f}）——"
    "如果這個數字明顯高於另外兩類，才值得投入 tie 專屬的處理；"
    "三類接近的話，代表損失是平均分散在三類，不是 tie 特別難。"
)
