# %% [markdown]
# # 訓練 DeBERTa-v3-base（里程碑 2 + 里程碑 3 的 5-fold 驗證）
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
# ## Label smoothing 實驗（milestone 3，已測完，結論：放棄）
#
# 在 fold 0 單獨測過 `label_smoothing=0.1`（output_dir 跟主要 5-fold 訓練分開，
# 不影響正式權重）：valid log loss 1.08815，比基準 1.06840 **變差 0.01975**——是
# TTA/校準實驗量到的雜訊量級（標準差 0.00288）的將近 7 倍，不是雜訊。放鬆訓練目標
# 的信心，讓模型在本來能有把握答對的題目上也不敢預測太肯定，log loss 對「答對但
# 不夠肯定」的懲罰蓋過了「答錯但太肯定」省下來的懲罰。結論：**不要用**，`train_fold()`
# 的 `label_smoothing` 參數留著（預設 0.0，行為不變），但不再需要重跑這個實驗。
#
# 詳細數字見 README.md「目前狀態」段落。milestone 3 剩下的候選手法：訓練時 a/b
# 對調增強、`winner_tie` 特殊處理。

# %% [markdown]
# ## `group_by_length` 實驗（訓練效能，先只在 fold 0 驗證，不直接動全部 5 folds）
#
# 白話說：把長度相近的樣本分到同一個 batch，減少「短句子被迫填充到跟長句子一樣長」
# 浪費掉的算力。獨立的 profiling kernel（`profile_train.py`）已經用真的 GPU 測過
# 最壞情況——把 fold 0 訓練集裡最長的 16 筆組成一個 batch，單步 forward+backward
# 沒有 OOM（記憶體餘裕 23.6%）。但那只驗證了「單一步驟不會炸」，這裡要驗證的是
# 完整一個 fold（2 epochs、45746 列）訓練下來：(1) 真的省了多少時間、(2) 分數
# 有沒有意外變差（理論上不該變差——只是改批次組成跟順序，模型看到的資料一樣多）、
# (3) 記憶體有沒有在長時間訓練後因為碎片化而撐不住（單步測試驗不出這個）。
#
# 比較基準：fold 0 目前（沒有 group_by_length）的訓練總耗時 6051 秒（約 1.68 小時）、
# valid log loss 1.06840。output_dir 跟主要的 5-fold 訓練分開，不會互相干擾，也
# 不會被下面的「跳過已存在權重」邏輯誤判成同一份。

# %%
import time as _time

FOLD0_BASELINE_SECONDS = 6051
FOLD0_BASELINE_LOG_LOSS = 1.06840

_t0 = _time.time()
gbl_result = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    group_by_length=True,
    output_dir="/kaggle/working/model/_group_by_length_fold0",
)
gbl_seconds = _time.time() - _t0
print(f"fold 0（group_by_length=True）訓練總耗時：{gbl_seconds:.0f}s（基準 {FOLD0_BASELINE_SECONDS}s，"
      f"{'省下' if gbl_seconds < FOLD0_BASELINE_SECONDS else '多花'} {abs(FOLD0_BASELINE_SECONDS - gbl_seconds):.0f}s，"
      f"{(FOLD0_BASELINE_SECONDS - gbl_seconds) / FOLD0_BASELINE_SECONDS * 100:+.1f}%）")
print(f"fold 0（group_by_length=True）valid log loss: {gbl_result['score']:.5f}")
print(f"fold 0 基準（無 group_by_length）: {FOLD0_BASELINE_LOG_LOSS:.5f}")
print(f"分數差異：{FOLD0_BASELINE_LOG_LOSS - gbl_result['score']:+.5f}（接近 0 才正常，明顯變差代表這個設定有問題）")
assert not gbl_result["diverged"] and gbl_result["n_nonfinite"] == 0, (
    "group_by_length 訓練發散或有非有限值（可能是長時間訓練下記憶體/數值比 profiling "
    "kernel 單步測試更緊繃），不要拿這個結果做決定，也不要直接套進下面的主要訓練迴圈"
)

# %% [markdown]
# **決定要不要繼續**：如果上面沒有發散、分數沒有明顯變差（在雜訊量級 0.003 以內）、
# 而且時間確實省下來了，值得把下面主要訓練迴圈的 5 個 `train_fold()` 呼叫都加上
# `group_by_length=True`。如果發散、OOM、或分數明顯變差，放棄這個設定，`train_fold()`
# 的 `group_by_length` 參數維持預設 False 不動。

# %% [markdown]
# 煙霧測試過關後才跑完整訓練。valid log loss 必須小於 1.09861（ln 3）—— 這是本專案
# 判斷分數的唯一標準，也是 scripts/baseline_prior.py 在真實資料上印出的基準
# （1.09723）。步進式存檔（每 1500 步）跟發散偵測保護都還在，不會再白燒一整個
# epoch。`eval_subset_rows=2000` 讓訓練中途的評估變快，最終回報的分數保證是對
# 完整驗證集算出來的。
#
# **這裡練滿全部 5 個 fold**——milestone 3 的 TTA + temperature scaling 只在 fold 0
# 單一份驗證集上量過改善（+0.00383），單一切分的量測有可能只是那份驗證集剛好對這個
# 手法有利，5 個 fold 各自獨立驗證同一個改善才有說服力。
#
# **實測踩過的坑（已修復）**：第一次跑 5 folds 時，`_training_args()` 的
# `output_dir` 跟 `trainer.save_model()` 的最終存檔目錄是同一個路徑，Trainer 自己
# `save_steps` 週期性存的 `checkpoint-*/`（含 optimizer/scheduler state，體積是
# 模型本身的 2-3 倍）一直沒清掉，5 個 fold 疊起來直接把 Kaggle 磁碟塞爆
# （`OSError: No space left on device`，練到 fold 4 過半才炸）。`train_fold()`
# 現在會在 `trainer.save_model()` 之後自動清掉 `output_dir` 底下的 `checkpoint-*/`，
# 每個 fold 只留最終權重（~700MB，不是 ~2.8GB）。
#
# 那次事故裡 fold 0-3 其實都順利練完、分數都贏過基準（1.06840 / 1.05461 / 1.07391 /
# 1.08028），只有 fold 4 沒存到——下面會先檢查 `/kaggle/input` 有沒有掛之前留下來的
# 權重，有的話直接複製過來、略過重新訓練，不用 5 個全部重練一次。

# %%
import pathlib
import shutil
import time

from llmcls.config import MODEL_DIR, N_FOLDS

# 如果有掛之前的訓練成果（例如上一次中途出錯，這次接續跑），先複製過來、跳過
# 已經練好的 fold，不用整個重來。找不到就當作全新開始，不影響正常流程。
#
# 檔名是攤平的 `fold{N}__檔名`（不是巢狀資料夾）——上傳 Kaggle Dataset 時，
# 巢狀資料夾要嘛跳過、要嘛整個壓成一個 zip/tar，`--dir-mode` 實際行為（會不會
# 自動解壓縮回資料夾）沒把握，攤平成單層檔名最保險，不用賭 Kaggle 那端怎麼處理。
for _p in pathlib.Path("/kaggle/input").glob("**/fold*__model.safetensors"):
    fold_name, _ = _p.name.split("__", 1)
    dst = MODEL_DIR / fold_name
    if not dst.exists():
        dst.mkdir(parents=True)
        for _f in _p.parent.glob(f"{fold_name}__*"):
            _, filename = _f.name.split("__", 1)
            shutil.copy(_f, dst / filename)
        print(f"{fold_name} 已從先前的權重複製過來，略過重新訓練")

# %%
fold_scores = {}
for fold in range(N_FOLDS):
    dst = MODEL_DIR / f"fold{fold}"
    if (dst / "model.safetensors").exists():
        print(f"fold {fold} 已經有存好的權重（{dst}），略過重新訓練")
        continue
    _t0 = time.time()
    result = train_fold(
        fold=fold, epochs=2, batch_size=8, max_len=512, lr=2e-5,
        fp16=True, eval_steps=1500, eval_subset_rows=2000,
    )
    print(f"fold {fold} 訓練總耗時：{time.time() - _t0:.0f}s")
    assert not result["diverged"] and result["n_nonfinite"] == 0, (
        f"fold {fold} 完整訓練發散或有非有限值，模型權重不可信，不要拿去推論"
    )
    fold_scores[fold] = result["score"]
    print(f"fold {fold} valid log loss: {result['score']:.5f}")

# %% [markdown]
# 每個 fold 都已經存到 `MODEL_DIR/fold{N}`（預設 `/kaggle/working/model/fold{N}`）。
#
# 確認全部贏過基準之後：**Save Version**（Save & Run All），存出的 Version 的
# `/kaggle/working/model/` 就會變成一個新的 Kaggle Dataset（在 Notebook 的 Output
# 分頁），下一步把它掛進 `calibrate_folds.py`（重新驗證 TTA/校準在 5 個 fold 上是否
# 一致改善）跟 `infer_deberta.py`。

# %%
print("五個 fold 的 valid log loss：")
for fold, score in fold_scores.items():
    print(f"  fold {fold}: {score:.5f}")
