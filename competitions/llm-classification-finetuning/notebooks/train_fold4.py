# %% [markdown]
# # 補練 fold 4，備份成獨立的 Kaggle Dataset
#
# **背景**：`train_kernel` 這幾次為了做 label smoothing / group_by_length 實驗，
# push 了好幾次「精簡版」notebook（冒煙測試 + 單一實驗，拿掉主要 5-fold 訓練），
# 每次成功跑完都會把 `train_kernel` 在 Kaggle 上「最新一次的輸出」整個換掉。
# 目前正式使用的 fold 0-4 權重（`infer_deberta.py` 送出 1.04529 那份）已經不在
# `train_kernel` 的最新輸出裡了——fold 0-3 還有安全網（`fold0-3-checkpoints`
# 這個獨立 Dataset），但 fold 4（valid log loss 1.05712 那份）目前完全抓不回來。
#
# 這個 kernel 是**獨立**的，不掛 `train_kernel` 的 `kernel_sources`、也不會被
# `train_kernel` 之後的 push 影響——只用跟原本 fold 4 一模一樣的設定（跟
# `train_deberta.py` 主要訓練迴圈相同：epochs=2、batch_size=8、lr=2e-5、
# fp16=True，**不加** `label_smoothing` 或 `group_by_length`——這兩個都已經
# 實測放棄，見 README.md）重新練一次 fold 4，練完直接攤平成
# `fold4__<檔名>` 的命名格式（跟 `fold0-3-checkpoints` 這個 Dataset 裡的檔名
# 格式一致），方便之後跟 fold 0-3 合併成一個涵蓋全部 5 folds 的新 Dataset。
#
# `add_folds`/`fold_indices` 是決定性切分，`TrainingArguments(seed=42)` 固定
# 訓練過程的隨機性，重跑應該會得到很接近 1.05712 的分數（GPU 運算本身有一點
# 非決定性，不保證位元級一致，但不該有肉眼可辨的差異）。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %% [markdown]
# 跟主要訓練迴圈用一模一樣的設定，唯一差別是只練 fold 4、輸出目錄用預設值
# （`MODEL_DIR/fold4`）。

# %%
from llmcls.train import train_fold

result = train_fold(
    fold=4, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
)
print(f"fold 4 valid log loss: {result['score']:.5f}  n_nonfinite={result['n_nonfinite']}  diverged={result['diverged']}")
assert not result["diverged"] and result["n_nonfinite"] == 0, (
    "fold 4 訓練發散或有非有限值，不要拿這份權重當正式權重用"
)
print("跟原本紀錄的 1.05712 比較：", f"{result['score'] - 1.05712:+.5f}")

# %% [markdown]
# 攤平成 `fold4__<檔名>` 格式，寫到 `/kaggle/working/checkpoints_flat/`——跟
# `fold0-3-checkpoints` Dataset 裡的檔名格式一致，下載這個 kernel 的輸出之後，
# 直接跟 fold 0-3 的檔案放在同一個資料夾就能組成完整的 5-fold Dataset，不用
# 再手動改檔名。

# %%
import shutil
from pathlib import Path

from llmcls.config import MODEL_DIR

flat_dir = Path("/kaggle/working/checkpoints_flat")
flat_dir.mkdir(parents=True, exist_ok=True)
fold4_dir = MODEL_DIR / "fold4"
for f in fold4_dir.iterdir():
    shutil.copy(f, flat_dir / f"fold4__{f.name}")
print("攤平後的檔案：", sorted(p.name for p in flat_dir.iterdir()))
