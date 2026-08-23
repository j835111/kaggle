# %% [markdown]
# # GPU 效能量測（獨立於訓練 kernel，不影響 train_deberta.py 正在跑的東西）
#
# 上一輪純研究的評估（讀程式碼，沒有實測）排出的優先順序：
#
# 1. `group_by_length=True`——預期效益最大，但風險反直覺：`batch_size=16` 曾在完整
#    45746 筆資料上因為「某一批全是長句子」CUDA OOM（只差 66MB）。`group_by_length`
#    做的事正是主動把長句子集中到同一批，等於把這個最壞情況從偶發變成每個 epoch
#    必然出現。這個 kernel 第一件事就是**直接組出那個最壞批次**（fold 0 訓練集裡最長
#    的 16 筆），跑一次真正的 forward+backward，看到底炸不炸。
# 2. `attn_implementation="sdpa"`——DeBERTa 的相對位置注意力機制歷史上不支援
#    PyTorch 的 SDPA/FlashAttention 加速，但沒查過這個 Kaggle 環境的 transformers
#    版本現況。這裡直接試載入、比較 eager vs sdpa 的每步耗時。
#
# 這個 kernel 用乾淨的 base model（不掛任何舊 checkpoint），只做量測，不存權重、
# 不產生 submission。

# %%
# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<

# %% [markdown]
# ## 量測 1：group_by_length 的最壞批次會不會 OOM
#
# 用 fold 0 真正的訓練切分，量出每一列組完 `[CLS] prompt [SEP] response_a [SEP]
# response_b [SEP]` 之後的真實序列長度（跟 `group_by_length` 排序時用的長度一致），
# 抓最長的 16 筆組成一個 batch —— 這就是 `group_by_length` 開啟後每個 epoch 一定會
# 出現的最壞批次，不是煙霧測試那種隨機抽樣可能抽不到的情況。

# %%
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llmcls.config import MAX_LEN, MODEL_NAME, N_CLASSES, SEED
from llmcls.cv import add_folds, fold_indices
from llmcls.data import load_train
from llmcls.text import build_input_ids

train = load_train()
train = add_folds(train)
tr_idx, _ = fold_indices(train, fold=0)
tr_df = train.iloc[tr_idx].reset_index(drop=True)
print(f"fold 0 train rows: {len(tr_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prompt_ids_all = tokenizer(tr_df["prompt_text"].tolist(), add_special_tokens=False)["input_ids"]
resp_a_ids_all = tokenizer(tr_df["response_a_text"].tolist(), add_special_tokens=False)["input_ids"]
resp_b_ids_all = tokenizer(tr_df["response_b_text"].tolist(), add_special_tokens=False)["input_ids"]

lengths = []
built_seqs = []
cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
for p_ids, a_ids, b_ids in zip(prompt_ids_all, resp_a_ids_all, resp_b_ids_all):
    p, a, b = build_input_ids(p_ids, a_ids, b_ids, MAX_LEN)
    seq = [cls_id, *p, sep_id, *a, sep_id, *b, sep_id]
    built_seqs.append(seq)
    lengths.append(len(seq))
lengths = np.array(lengths)

print(
    f"序列長度分佈：min={lengths.min()} p50={int(np.percentile(lengths, 50))} "
    f"p90={int(np.percentile(lengths, 90))} p99={int(np.percentile(lengths, 99))} max={lengths.max()}"
)
print(f"打滿 max_len({MAX_LEN}) 的比例：{(lengths == MAX_LEN).mean():.4f}")

worst_idx = np.argsort(-lengths)[:16]
worst_seqs = [built_seqs[i] for i in worst_idx]
worst_lengths = [int(lengths[i]) for i in worst_idx]
print(f"最長 16 筆的長度：{worst_lengths}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=N_CLASSES)
model = model.float().cuda()
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scaler = torch.cuda.amp.GradScaler()

max_len_in_batch = max(worst_lengths)
pad_id = tokenizer.pad_token_id or 0
input_ids = torch.full((16, max_len_in_batch), pad_id, dtype=torch.long)
attention_mask = torch.zeros((16, max_len_in_batch), dtype=torch.long)
for i, seq in enumerate(worst_seqs):
    input_ids[i, : len(seq)] = torch.tensor(seq)
    attention_mask[i, : len(seq)] = 1
labels = torch.zeros(16, dtype=torch.long)

input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

oom_result = {"oom": False, "detail": None}
try:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("group_by_length 最壞批次（batch_size=16，全部最長序列）forward+backward 成功")
    print(f"peak memory allocated: {peak:.3f} GiB / total {total:.3f} GiB，餘裕 {total - peak:.3f} GiB")
    oom_result = {"oom": False, "peak_gib": round(peak, 3), "total_gib": round(total, 3)}
except torch.cuda.OutOfMemoryError as e:
    print("group_by_length 最壞批次 CUDA OOM：", str(e))
    oom_result = {"oom": True, "detail": str(e)}

print("OOM_RESULT:", oom_result)

del model, optimizer, scaler
torch.cuda.empty_cache()

# %% [markdown]
# ## 量測 2：`attn_implementation="sdpa"` 支不支援，速度差多少
#
# DeBERTa 的相對位置注意力是自訂運算，歷史上不一定吃得到 PyTorch 的
# `scaled_dot_product_attention` 加速。先試載入，`model.config._attn_implementation`
# 會告訴我們 transformers 實際用了哪個實作（要求 sdpa 但環境不支援的話，通常會靜默
# fallback 成 eager，不會直接報錯，所以不能只看有沒有丟例外）。

# %%
sdpa_supported = True
sdpa_error = None
try:
    model_sdpa_probe = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, attn_implementation="sdpa"
    )
    actual = model_sdpa_probe.config._attn_implementation
    print("要求 sdpa，實際採用：", actual)
    if actual != "sdpa":
        sdpa_supported = False
        sdpa_error = f"靜默 fallback 成 {actual}"
    del model_sdpa_probe
except Exception as e:  # noqa: BLE001 —— 這裡要老實記錄任何失敗原因，不分類型
    sdpa_supported = False
    sdpa_error = repr(e)

print("SDPA_SUPPORTED:", sdpa_supported, "detail:", sdpa_error)

# %% [markdown]
# ## 量測 3：eager vs sdpa 的實際訓練速度 + CPU/GPU 時間分佈
#
# 用 fold 0 訓練集隨機抽 2000 筆（跟正式訓練同樣的 tokenize + 截斷邏輯），各跑
# 15 步 forward+backward，比較平均每步耗時、peak memory，並用 `torch.profiler`
# 印出耗時最高的 15 個 op —— 用來判斷 CUDA 時間佔比多少、有沒有明顯的 CPU-bound
# （例如 tokenize、DataLoader 相關 op 佔比異常高）訊號。

# %%
import torch.profiler as profiler
from transformers import DataCollatorWithPadding

subset = tr_df.sample(n=min(2000, len(tr_df)), random_state=SEED).reset_index(drop=True)
BATCH_SIZE, N_STEPS = 8, 15


def make_batches(df, tokenizer, max_len, batch_size, n_steps):
    p_ids = tokenizer(df["prompt_text"].tolist(), add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(df["response_a_text"].tolist(), add_special_tokens=False)["input_ids"]
    b_ids = tokenizer(df["response_b_text"].tolist(), add_special_tokens=False)["input_ids"]
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    n_needed = min(len(df), batch_size * n_steps)
    items = []
    for i in range(n_needed):
        p, a, b = build_input_ids(p_ids[i], a_ids[i], b_ids[i], max_len)
        seq = [cls_id, *p, sep_id, *a, sep_id, *b, sep_id]
        items.append({"input_ids": seq, "attention_mask": [1] * len(seq), "labels": int(df["label"].iloc[i])})
    batches = [collator(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]
    return batches[:n_steps]


def run_profiled(model, batches, label):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        profile_memory=True,
    ) as prof:
        for batch in batches:
            batch = {k: v.cuda() for k, v in batch.items()}
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    ms_per_step = elapsed / len(batches) * 1000
    print(f"=== {label} ===")
    print(f"{len(batches)} steps，總耗時 {elapsed:.2f}s，平均每步 {ms_per_step:.1f}ms，peak memory {peak:.3f} GiB")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return {"label": label, "elapsed_s": round(elapsed, 3), "ms_per_step": round(ms_per_step, 1), "peak_gib": round(peak, 3)}


batches = make_batches(subset, tokenizer, MAX_LEN, BATCH_SIZE, N_STEPS)

model_eager = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=N_CLASSES).float().cuda()
eager_result = run_profiled(model_eager, batches, "eager")
del model_eager
torch.cuda.empty_cache()

sdpa_result = None
if sdpa_supported:
    model_sdpa = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, attn_implementation="sdpa"
    ).float().cuda()
    sdpa_result = run_profiled(model_sdpa, batches, "sdpa")
    del model_sdpa
    torch.cuda.empty_cache()
    diff_pct = (eager_result["ms_per_step"] - sdpa_result["ms_per_step"]) / eager_result["ms_per_step"] * 100
    print(f"sdpa vs eager 每步耗時差異：{diff_pct:+.1f}%（正值代表 sdpa 更快）")
else:
    print("sdpa 不支援，跳過比較")

# %% [markdown]
# ## 總結（三個量測結果都印在這裡，方便事後從 log 一次抓完）

# %%
print("========== FINAL_SUMMARY ==========")
print("OOM_RESULT:", oom_result)
print("SDPA_SUPPORTED:", sdpa_supported, "detail:", sdpa_error)
print("EAGER_RESULT:", eager_result)
print("SDPA_RESULT:", sdpa_result)
print("====================================")
