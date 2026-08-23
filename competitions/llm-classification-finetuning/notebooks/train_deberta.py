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
# ## Label smoothing 實驗（milestone 3，重新做一次——第一次測的方法有漏洞）
#
# **第一次測的問題**：在補練 fold 4 的時候發現，`train_fold()` 原本在載入模型
# （`AutoModelForSequenceClassification.from_pretrained()`，分類頭是隨機初始化的
# 新的一層）之後才建立 `Trainer`、`TrainingArguments(seed=42)` 才生效——分類頭的
# 起始隨機值從來沒被這個 seed 固定住。實測同一個 fold、完全相同設定重跑一次，valid
# log loss 可以飄動 0.01~0.02，跟第一次判定 label smoothing 「變差 0.01975」是同一
# 個量級——那次比較的兩個分數來自不同時間跑的兩次訓練，分類頭起始值本來就不一樣，
# 差距有可能大部分只是隨機運氣，不是 label smoothing 真的有害。已經在 `train_fold()`
# 裡加了 `set_seed(SEED)`（在載入模型之前呼叫），現在同一個 seed 重跑會拿到同一個
# 分類頭起始值。
#
# **這次重做**：baseline 跟 `label_smoothing=0.1` 都在**同一次 kernel 執行**裡各自
# 呼叫一次 `train_fold()`——兩次呼叫各自呼叫到的 `set_seed(SEED)` 保證兩邊的分類頭
# 起始值一致，才是真正公平的 A/B 比較，不再跟舊 session 的歷史數字比較。output_dir
# 都跟主要 5-fold 訓練分開，不影響正式權重。

# %%
ab_baseline = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    output_dir="/kaggle/working/model/_seed_fixed_baseline_fold0",
)
print(f"baseline（無 label smoothing）valid log loss: {ab_baseline['score']:.5f}")
assert not ab_baseline["diverged"] and ab_baseline["n_nonfinite"] == 0, (
    "baseline 重跑就發散了，seeding 修法本身可能有問題，先不要信下面的比較"
)

ab_label_smoothing = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    label_smoothing=0.1,
    output_dir="/kaggle/working/model/_seed_fixed_label_smoothing_fold0",
)
print(f"label_smoothing=0.1 valid log loss: {ab_label_smoothing['score']:.5f}")
assert not ab_label_smoothing["diverged"] and ab_label_smoothing["n_nonfinite"] == 0, (
    "label smoothing 訓練發散或有非有限值，不要拿這個結果做決定"
)

print(f"差異：{ab_baseline['score'] - ab_label_smoothing['score']:+.5f}（正值代表 label smoothing 有幫助）")

# %% [markdown]
# 詳細數字見 README.md「目前狀態」段落。milestone 3 剩下的候選手法：訓練時 a/b
# 對調增強、`winner_tie` 特殊處理。

# %% [markdown]
# ## `group_by_length` 實驗（訓練效能，已測完，結論：放棄）
#
# 在 fold 0 單獨測過（output_dir 跟主要 5-fold 訓練分開，不影響正式權重）：訓練在
# epoch 0.4809（原訂 2 epochs，只跑了 24%）就被 `StopOnNonFiniteLoss` 攔截停止——
# `grad_norm` 在那一步變成 `inf`，跟這個模型已知在 fp32/peak LR 附近容易發散是
# 同一類問題（見里程碑 2 的踩坑紀錄）。合理推測：`group_by_length` 把長度相近的
# 樣本集中到同一個 batch，可能讓某些 batch 全部是長句子、梯度震幅比原本長短混合的
# batch 更劇烈，在這個已知敏感的模型上更容易踩到 fp16 數值溢位。
#
# **這裡踩到訓練實驗的一個通用陷阱要記錄**：訓練被安全機制提前攔停之後，記錄下來的
# 「耗時 1256s（比基準 6051s 快 79%）」跟「valid log loss 1.09921（比基準 1.06840
# 差 0.03081）」**兩個數字都不是有效比較**——不是 group_by_length 讓訓練變快，是
# 訓練只跑了四分之一就被緊急煞車攔下來；分數變差也是因為模型根本沒練完，不是
# group_by_length 本身讓分數變爛。判斷一個訓練實驗有沒有意義，第一步永遠是先確認
# `diverged=False`，時間和分數的比較才有意義。
#
# **註記**：這次測試是在發現分類頭初始化沒被 seed 固定（見上面 label smoothing
# 那段）之前做的，理論上不同的隨機初始值也可能影響訓練在哪個點對梯度爆炸比較敏感。
# 但這裡是**直接發散**（grad_norm 變 inf），不是「分數飄動 0.01~0.02」這種量級的
# 問題，用不同初始值再測一次也未必會每次都發散，但發散本身仍然是一個真實訊號，不是
# 憑空捏造的——只是還沒有用修好 seeding 的版本重新驗證過，結論維持放棄，但信心
# 沒有 label smoothing 那次（已經重測過）那麼高。
#
# 結論：**不要用**，`train_fold()` 的 `group_by_length` 參數留著（預設 False，
# 行為不變）。要讓它可用可能需要額外調低 peak LR 或拉長 warmup，但這是額外的調參
# 投入，跟 label smoothing 一樣先不追加投入，把力氣留給 milestone 3 剩下的候選
# 手法。詳細數字見 README.md「目前狀態」段落。

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
