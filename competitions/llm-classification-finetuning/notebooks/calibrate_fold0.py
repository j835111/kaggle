# %% [markdown]
# # 里程碑 3：TTA（a/b 對調）與 temperature scaling 校準實驗
#
# 這是**實驗量測**用的 notebook，不重新訓練，只對 fold 0 已經存好的權重、在它自己的
# 驗證集（`fold_indices(fold=0)` 切出來的 held-out 那份，訓練時從沒看過）上量測四種
# 組合的 log loss，決定哪個組合值得真的套進 `infer_deberta.py`：
#
# 1. baseline —— 原始順序，沒有 TTA、沒有校準（milestone 2 送出去的排行榜就是這個）
# 2. TTA —— response_a/response_b 對調各推論一次，換回欄位對齊後在機率層級平均，
#    對付「模型偏好某個位置本身」的位置偏誤
# 3. temperature scaling —— 用同一份驗證集的 logits 配一個純量溫度 T
#    （只有一個參數，held-out 資料上配它不算作弊，是標準做法）
# 4. TTA + temperature —— 先把兩個順序的 logits 在 logits 層級平均（校準要對的是
#    logits 的尺度，不是已經 softmax 過的機率），再對合併後的 logits 配溫度
#
# 額外量測一項診斷：只看「對調順序、換回欄位對齊」單獨的 log loss（不平均），
# 拿來跟 baseline 比——差距大代表模型真的有位置偏誤，TTA 值得做；差距小代表
# 位置偏誤不明顯，校準才是重點。
#
# 用 `kaggle kernels push` 上傳（見 notebooks/README.md 的模式）：`competition_sources`
# 掛這場競賽的資料（重建 fold 0 的驗證切分要用到 train.csv），`kernel_sources` 接
# 訓練 kernel 的輸出（讀 fold0 checkpoint），**關網路**（跟推論一樣，讀本地權重、
# 不用連 HuggingFace Hub）。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %% [markdown]
# **不要猜掛載路徑**——跟 `infer_deberta.py` 一樣的坑，直接 glob 找 fold0 checkpoint
# 實際在哪。

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

# %% [markdown]
# 重建 fold 0 的驗證切分——要跟 `train_fold(fold=0)` 訓練時用的是同一份 held-out
# 資料，才能公平比較。`add_folds` / `fold_indices` 是同一套決定性切分邏輯，seed
# 跟 n_folds 都用預設值，重跑一次會得到一模一樣的切分。

# %%
from llmcls.calibration import apply_temperature, fit_temperature
from llmcls.config import MAX_LEN, MODEL_DIR, N_FOLDS
from llmcls.cv import add_folds, fold_indices
from llmcls.data import load_train
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss, softmax
from llmcls.train import load_trained, predict_logits_with_model, swap_ab
from llmcls.tta import average_swapped

train = load_train()
train = add_folds(train, n_folds=N_FOLDS)
_, va_idx = fold_indices(train, fold=0)
va_df = train.iloc[va_idx].reset_index(drop=True)
labels = va_df["label"].to_numpy()
print(f"fold 0 驗證集：{len(va_df)} 列")

model, tokenizer = load_trained(MODEL_DIR)

# %% [markdown]
# 兩次推論：原始順序、對調順序。都只推論一次、存下原始 logits，後面的四種組合全部
# 從這兩組 logits 算出來，不重複跑 forward——GPU 時間不用浪費在同一份資料上重複推論。

# %%
logits_orig = predict_logits_with_model(model, tokenizer, va_df, max_len=MAX_LEN, batch_size=32)
logits_swapped = predict_logits_with_model(model, tokenizer, swap_ab(va_df), max_len=MAX_LEN, batch_size=32)

probs_orig = softmax(logits_orig)
# 對調後模型輸出的欄位是「目前放在 A 位置的贏 / 目前放在 B 位置的贏 / tie」，
# 但「目前 A 位置」其實是原本的 response_b，換回來才能跟 probs_orig 對齊比較。
probs_swapped_aligned = softmax(logits_swapped)[:, [1, 0, 2]]

# %% [markdown]
# 四種組合 + 一項位置偏誤診斷。

# %%
baseline_loss = log_loss(labels, probs_orig)
swapped_only_loss = log_loss(labels, probs_swapped_aligned)

tta_probs = average_swapped(probs_orig, softmax(logits_swapped))
tta_loss = log_loss(labels, tta_probs)

t_baseline = fit_temperature(logits_orig, labels)
temp_probs = apply_temperature(logits_orig, t_baseline)
temp_loss = log_loss(labels, temp_probs)

combined_logits = average_swapped(logits_orig, logits_swapped)
t_combined = fit_temperature(combined_logits, labels)
tta_temp_probs = apply_temperature(combined_logits, t_combined)
tta_temp_loss = log_loss(labels, tta_temp_probs)

print(f"均勻亂猜基準（ln 3）        {UNIFORM_LOGLOSS:.5f}")
print(f"baseline（milestone 2）     {baseline_loss:.5f}")
print(f"單獨對調順序（診斷用）      {swapped_only_loss:.5f}  跟 baseline 差 {swapped_only_loss - baseline_loss:+.5f}")
print(f"TTA（a/b 對調平均）         {tta_loss:.5f}  改善 {baseline_loss - tta_loss:+.5f}")
print(f"temperature scaling（T={t_baseline:.3f}）  {temp_loss:.5f}  改善 {baseline_loss - temp_loss:+.5f}")
print(f"TTA + temperature（T={t_combined:.3f}）    {tta_temp_loss:.5f}  改善 {baseline_loss - tta_temp_loss:+.5f}")

print()
bias_gap = abs(swapped_only_loss - baseline_loss)
if bias_gap > 0.01:
    print(f"位置偏誤明顯（差距 {bias_gap:.5f} > 0.01）——TTA 應該有實質幫助")
else:
    print(f"位置偏誤不明顯（差距 {bias_gap:.5f} <= 0.01）——TTA 效果有限，校準才是重點")

results = {
    "baseline": baseline_loss,
    "swapped_only": swapped_only_loss,
    "tta": tta_loss,
    "temperature": temp_loss,
    "tta_temperature": tta_temp_loss,
}
best_name = min(results, key=results.get)
print(f"\n最佳組合：{best_name}（{results[best_name]:.5f}）—— 把這個套進 infer_deberta.py")
