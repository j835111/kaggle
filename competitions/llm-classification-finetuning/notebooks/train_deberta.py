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
# ## 決定性檢查（診斷用，已測完，結論：修復有效）
#
# a/b 對調增強實驗測出的效果量（+0.00899）跟兩次不同 kernel 執行的 baseline
# 落差（0.01226）同一個量級，追查發現 `train_fold()` 有兩個會讓「同樣設定重跑」
# 得到不同分數的量測 bug（詳細機制見 CLAUDE.md「兩個會讓 A/B 比較失去意義的量測
# bug」段落）：
#
# 1. `StopOnNonFiniteLoss` 誤判——單一次 log 看到 `grad_norm` 非有限值就喊停，
#    但那其實是 fp16 GradScaler 正常的「跳過這步、調低 scale 繼續」，不是真的
#    訓練壞掉（group_by_length 那次「發散」就是這樣被誤判的）。
# 2. `load_best_model_at_end` 靠訓練中途一個 2000 筆小樣本子集的分數去挑「最佳」
#    checkpoint，這個子集雜訊夠大，同樣設定的兩次訓練各自挑到不同進度的
#    checkpoint 當最終權重。
#
# 兩個都修好之後（`grad_norm` 要連續 3 次非有限值才停；`load_best_model_at_end`
# 關掉，固定用訓練跑完的最終狀態），同一次 kernel 執行內重跑兩次完全相同設定：
# **1.08448 vs 1.08683，差異只有 -0.00235**——比雜訊量級（標準差 0.00288）還小，
# 回到合理範圍。**同一次 kernel 執行內的 A/B 比較重新可信**，下面用這個修好的
# 版本重測 label smoothing 跟 group_by_length。

# %% [markdown]
# ## 重測 label smoothing + group_by_length（用修好的量測方法，共用一個 baseline）
#
# 這兩個手法之前的「放棄」結論都是在上面兩個量測 bug 修好之前測的，信心度低：
# label smoothing 第一次判定「變差 0.01975」，後來發現分類頭初始值沒固定住，
# 重測後差異只剩 +0.00049（雜訊量級）；group_by_length 判定「發散」，後來查出
# 觸發停止那一行 log 其實 loss 健康（1.079）、只有 grad_norm 是 inf，跟
# `StopOnNonFiniteLoss` 誤判的訊號一模一樣，很可能從來沒有真的發散過。
#
# 這次同一個 kernel 執行內共用**一個** baseline，分別跟 `label_smoothing=0.1`、
# `group_by_length=True` 比——兩個手法互不相關，用同一個 baseline 比對是合理的，
# 不需要各自重跑一次 baseline，省下一次 fold 0 訓練的時間。`group_by_length`
# 的記憶體風險（把長句子集中到同一個 batch）之前只測過單步 forward+backward
# 沒有 OOM，這次是完整 2 epochs 訓練，`diverged`/`n_nonfinite` 檢查不放寬。

# %%
retest_baseline = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    output_dir="/kaggle/working/model/_retest_baseline_fold0",
)
print(f"baseline valid log loss: {retest_baseline['score']:.5f}")
assert not retest_baseline["diverged"] and retest_baseline["n_nonfinite"] == 0, (
    "baseline 訓練發散或有非有限值，不要拿這個結果做決定"
)

retest_label_smoothing = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    label_smoothing=0.1,
    output_dir="/kaggle/working/model/_retest_label_smoothing_fold0",
)
print(f"label_smoothing=0.1 valid log loss: {retest_label_smoothing['score']:.5f}")
assert not retest_label_smoothing["diverged"] and retest_label_smoothing["n_nonfinite"] == 0, (
    "label smoothing 訓練發散或有非有限值，不要拿這個結果做決定"
)
print(
    f"label smoothing 差異：{retest_baseline['score'] - retest_label_smoothing['score']:+.5f}"
    "（正值代表 label smoothing 有幫助）"
)

retest_group_by_length = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    group_by_length=True,
    output_dir="/kaggle/working/model/_retest_group_by_length_fold0",
)
print(f"group_by_length=True valid log loss: {retest_group_by_length['score']:.5f}")
assert not retest_group_by_length["diverged"] and retest_group_by_length["n_nonfinite"] == 0, (
    "group_by_length 訓練發散或有非有限值——這次是修好誤判邏輯之後還發散，這次是真的，不要拿這個結果做決定"
)
print(
    f"group_by_length 差異：{retest_baseline['score'] - retest_group_by_length['score']:+.5f}"
    "（正值代表 group_by_length 有幫助）"
)

# %% [markdown]
# 詳細數字見 README.md「目前狀態」段落。

# %% [markdown]
# ## 訓練時 a/b 對調增強實驗（milestone 3，這次要測的）
#
# **動機**：`calibrate_folds.py` 的 winner_tie 診斷（把 5 個 fold 的驗證集預測
# 拼起來、照真實答案分三類算 log loss）發現模型對 `winner_model_a`/`winner_model_b`
# 兩類原始（未做 TTA/校準）的 log loss 落差很大：1.04710 vs 1.11257——反而
# `winner_tie` 本身沒有特別難（1.05011，三類裡最低）。推論時的 TTA（a/b 對調再
# 平均）已經在事後修正這個落差的一部分，這裡要測的是「直接在訓練時解決」：訓練
# 時讓模型有機率看到對調過順序（連同標籤一起對調：`swap_ab_label()`，
# `winner_model_a`⟷`winner_model_b` 互換、`winner_tie` 不變）的版本，減少模型
# 對「誰先出現」的偏見，而不是只靠推論時事後補救。
#
# **實作**：`PreferenceDataset` 新增 `ab_swap_prob` 參數——每一筆資料在每次
# `__getitem__` 被抓取時（也就是每個 epoch）都重新擲一次骰子，不是固定對調某
# 一半資料，同一列在不同 epoch 可能拿到不同順序。只套用在訓練集，驗證集固定用
# 原始順序不受影響——`score` 才能跟沒開這個選項的訓練直接比較。用獨立的
# `np.random.default_rng(SEED)`，不動 Trainer 自己靠全域亂數狀態做的資料洗牌。
# `swap_ab_label()`（`llmcls/text.py`，純 Python、本機已經有測試）確保標籤對調
# 邏輯不會弄錯方向。
#
# **這次測試**：比照 label smoothing 修好種子後的做法，baseline 跟
# `ab_swap_prob=0.5` 都在同一次 kernel 執行裡各自呼叫一次 `train_fold()`，分類頭
# 起始值保證一致才是公平 A/B。先跑一個小規模煙霧測試確認新程式碼路徑在 GPU 上不會
# 直接壞掉（本機沒有 torch，這段程式碼完全沒在真正的 GPU/tokenizer 上跑過），
# 通過才進入正式的 fold 0 控制實驗（約 4 小時 GPU）。
#
# **需要留意的風險**：TTA 已經在吃掉一部分「順序偏見」的紅利，這次的訓練時增強
# 有可能只是重複做同一件事、疊加起來看不出額外差異，也有可能真的讓模型本身更不
# 偏頗、帶來 TTA 吃不到的額外改善——這是實測才知道的事，不是先驗判斷得出來的。

# %%
ab_swap_smoke = train_fold(
    fold=0, epochs=1, batch_size=8, max_len=512, lr=2e-5,
    lr_scheduler_type="constant_with_warmup", fp16=True,
    max_train_rows=3000, max_valid_rows=800, max_steps=100, eval_steps=50,
    ab_swap_prob=0.5,
    output_dir="/kaggle/working/model/_ab_swap_smoke",
)
print(f"ab_swap_prob 煙霧測試 log loss={ab_swap_smoke['score']:.5f}  diverged={ab_swap_smoke['diverged']}")
assert not ab_swap_smoke["diverged"] and ab_swap_smoke["n_nonfinite"] == 0, (
    "ab_swap_prob 新程式碼路徑煙霧測試就不穩定，不要繼續跑正式實驗"
)
print("煙霧測試通過，ab_swap_prob 這條新程式碼路徑沒有明顯壞掉")

# %%
ab_swap_baseline = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    output_dir="/kaggle/working/model/_ab_swap_baseline_fold0",
)
print(f"baseline（無 a/b 對調增強）valid log loss: {ab_swap_baseline['score']:.5f}")
assert not ab_swap_baseline["diverged"] and ab_swap_baseline["n_nonfinite"] == 0, (
    "baseline 訓練發散或有非有限值，不要拿這個結果做決定"
)

ab_swap_variant = train_fold(
    fold=0, epochs=2, batch_size=8, max_len=512, lr=2e-5,
    fp16=True, eval_steps=1500, eval_subset_rows=2000,
    ab_swap_prob=0.5,
    output_dir="/kaggle/working/model/_ab_swap_variant_fold0",
)
print(f"ab_swap_prob=0.5 valid log loss: {ab_swap_variant['score']:.5f}")
assert not ab_swap_variant["diverged"] and ab_swap_variant["n_nonfinite"] == 0, (
    "a/b 對調增強訓練發散或有非有限值，不要拿這個結果做決定"
)

print(f"差異：{ab_swap_baseline['score'] - ab_swap_variant['score']:+.5f}（正值代表 a/b 對調增強有幫助）")

# %% [markdown]
# 詳細數字見 README.md「目前狀態」段落。milestone 3 剩下的候選手法：`winner_tie`
# 特殊處理——今天的診斷顯示 tie 反而不是損失最集中的類別，優先度已下修。

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
#
# **這次要正式套用 `label_smoothing=0.1`**（見上面的重測結果，同一次 kernel 執行
# 內控制實驗量到 +0.03341，遠大於雜訊量級，值得整批重練——見 CLAUDE.md「Label
# smoothing / 訓練時資料增強這類手法，改了就得整批重練」）。`fold0-4-checkpoints`
# 這個 Dataset 裡的 5 個 fold 都是**沒有** label smoothing 練出來的舊權重，訓練
# 配方換了，不能沿用、必須全部重練——`train_kernel/kernel-metadata.json` 的
# `dataset_sources` 已經清空，下面的複製-跳過邏輯不會找到任何東西可跳過，5 個
# fold 都會真的重新訓練。訓練完成後務必比照上次的做法，把輸出打包成新的 Dataset
# 取代 `fold0-4-checkpoints`，`train_kernel`/`infer_kernel`/`calibrate_kernel`
# 的 `dataset_sources` 都要跟著改指到新 Dataset。

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
        label_smoothing=0.1,
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
