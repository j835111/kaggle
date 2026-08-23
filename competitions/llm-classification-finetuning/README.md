# LLM Classification Finetuning

Kaggle 競賽 [llm-classification-finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)
的工作區。任務：給定一個 prompt 與兩個 LLM 的回覆，預測人類評審偏好哪一邊（三分類：
`winner_model_a` / `winner_model_b` / `winner_tie`），評分指標為 multi-class **log loss**。

**判斷分數的唯一標準：均勻亂猜 = ln(3) ≈ 1.0986。** 沒打敗這條線的模型等於沒有資訊量。

參考點：曾有參賽者用 TF-IDF + 邏輯迴歸（`max_features=150`）拿到 1.116，比亂猜還差。
這是單一一次的初版提交、不是該方法的上限，但方向很明確 —— 這題得真的做語言模型微調，
傳統特徵工程那條路走不通。

## 為什麼是模組而不是單一 notebook

本機沒有 GPU，訓練會在 Kaggle Notebook 或遠端 GPU 上跑。所以資料解析、CV 切分、評分、
提交組裝寫成 `src/llmcls/` 下可 import 的模組，兩邊共用同一份程式碼；路徑差異全部由
`llmcls/config.py` 吸收（偵測到 `/kaggle/input/` 就自動切換）。

## 目錄結構

```
src/llmcls/
  config.py      路徑、欄位名、seed —— 本機 vs Kaggle 的差異只在這裡
  data.py        CSV 載入 + JSON 多輪對話欄位解析 + schema 斷言
  cv.py          以 prompt 分組、以 label 分層的 StratifiedGroupKFold
  metrics.py     log loss（自行實作以確保與 Kaggle 算法一致）
  submission.py  提交檔組裝與格式驗證
  text.py        截斷邏輯（head+tail、三段預算分配）—— 純 Python，不碰 torch
  train.py       DeBERTa 微調：Dataset 組裝、train_fold()、推論 —— 需要 torch/transformers
scripts/
  download_data.sh   下載競賽資料（需先備妥憑證，見下）
  make_fixture.py    產生合成 fixture，供無資料時跑通 pipeline
  baseline_prior.py  里程碑 0：類別先驗 baseline
  train.py           里程碑 2：DeBERTa 微調的 CLI 入口（遠端 GPU 用；Kaggle Notebook 建議直接 import llmcls.train）
tests/             本機 pipeline 的單元測試（不碰 torch，含 text.py 的截斷邏輯）
notebooks/         Kaggle Notebook 草稿（train_deberta.py 訓練 / infer_deberta.py 推論，見 notebooks/README.md）
data/              競賽資料（git 忽略）
outputs/           提交檔與產出（git 忽略）
```

## 快速開始

以下指令都**在本競賽目錄內執行**（`--data-dir` 之類的參數是相對於 cwd 解析的）。
Python 直譯器用工作區根目錄的共用 venv，所以路徑是 `../../.venv/`。

```bash
cd competitions/llm-classification-finetuning

# 首次：在工作區根目錄建共用 venv（已存在就跳過這行 —— uv venv 會直接重建掉它）
(cd ../.. && uv venv)

# 裝本競賽的相依；--python 不能省，否則 uv 會去找當前目錄下不存在的 .venv
uv pip install --python ../../.venv/bin/python -r requirements.txt

# 無需真實資料即可跑通整條 pipeline
../../.venv/bin/python scripts/make_fixture.py
../../.venv/bin/python scripts/baseline_prior.py --data-dir data/fixture
../../.venv/bin/python -m pytest tests/ -q
```

## 下載競賽資料

尚未完成，需要你手動處理兩件事：

1. 到[競賽規則頁](https://www.kaggle.com/competitions/llm-classification-finetuning/rules)按下
   **I Understand and Accept**。沒接受規則的話 API 一律回 403。
2. 認證擇一：
   - 舊版：到 <https://www.kaggle.com/settings> 產生 API token，存成 `~/.kaggle/kaggle.json`
     並 `chmod 600`。
   - 新版（kaggle CLI 2.x）：執行 `kaggle login`（瀏覽器登入），token 會存在
     `~/.kaggle/access_token`，`download_data.sh` 兩種都認。

然後（一樣在本競賽目錄內）：

```bash
bash scripts/download_data.sh
../../.venv/bin/python scripts/baseline_prior.py    # 這次跑真實資料
```

## 目前狀態

已驗證（真實資料，57,477 列訓練集，24 項測試全綠）：

- JSON 多輪對話欄位解析，含 null 與非法 JSON 的退路
- 以 prompt 分組的 5-fold 切分（51,443 個唯一 prompt），並斷言無跨 fold 洩漏
- log loss 計算，均勻預測恰好等於 ln(3)
- 提交檔組裝與格式驗證（欄位、機率和、重複 id、與 sample_submission 的 id 集合比對）
- **欄位格式**：`prompt` / `response_a` / `response_b` 確認為 JSON 編碼的字串陣列，與推定
  的 LMSYS schema 一致。類別先驗 baseline 在真實資料上 OOF log loss = 1.09723
  （基準 ln(3) = 1.09861）。
- `llmcls/text.py` 的截斷邏輯（三段預算分配、head+tail、邊界情況）—— 純 Python，不用
  真的 tokenizer 就能測。

尚未驗證：

已在 Kaggle 上實測驗證（GPU T4，`kaggle kernels push` 自動觸發，見
`notebooks/README.md`）：

- fold 0、DeBERTa-v3-base、2 epochs、lr=2e-5、linear decay：**valid log loss
  1.08494**（基準 1.09861，改善 +0.01367，`n_nonfinite: 0`，全程無發散），
  **訓練總耗時 6051 秒（約 1.68 小時）**——第一次跑通花了將近 9 小時，靠正規
  fp16 混合精度（`fp16=True`，`model.float()` 之後才啟用，是有 GradScaler
  保護的真混合精度，跟除錯過程中踩到的裸 fp16 不同）+ 訓練中途評估改用子集
  （`eval_subset_rows`，最終分數仍是對完整驗證集算的）兩項加速，快了約 5.4 倍。
  `batch_size` 也試過 8→16：煙霧測試（3000 筆子集）完全穩定，換成完整
  45746 筆卻在某批全是長序列時 CUDA OOM（只差 66MB）——煙霧測試驗得出穩定性，
  驗不出完整資料集的記憶體上限，所以維持 `batch_size=8`。
- 實測踩到的坑，修復都在 `llmcls/train.py` / `llmcls/config.py` 裡：
  - `competition_sources` 透過 API push 掛載時的路徑是
    `/kaggle/input/competitions/<slug>/`，不是網頁 UI 掛資料集的
    `/kaggle/input/<slug>/`。
  - Kaggle 沒指定 `machine_shape` 會自動配到 P100，跟目前的 torch 版本（只支援
    sm_70+）不相容，必須指定 `NvidiaTeslaT4`。
  - **關鍵**：`microsoft/deberta-v3-base` 在 HF Hub 上是用 fp16 存的，
    `AutoModelForSequenceClassification.from_pretrained()` 預設照抄 checkpoint
    原本的 dtype，跟 `TrainingArguments(fp16=False)` 無關——等於一直在跑沒有
    loss scaler 保護的裸 fp16 訓練，訓練到一半必然 NaN。修法是載入後強制
    `.float()`。
  - `kernel_sources` 的掛載路徑一樣不能用猜的（跟 `competition_sources` 同一類
    坑）：實測掛在 `/kaggle/input/notebooks/<owner>/<kernel-slug>/...`，`infer_
    deberta.py` 改成用 `pathlib.glob` 動態找 checkpoint，不寫死路徑。

離線推論 notebook（`infer_deberta.py`）也已經在 Kaggle 上跑通：關網路、讀訓練
notebook 存出的 fp32 權重、對 `test.csv` 推論、`validate_submission()` 通過、寫出
`submission.csv`，並已經送到排行榜：**143/212**（第一次送出、單一 fold、沒有
TTA / 校準 / ensemble）。

**里程碑 3（TTA + 校準）已在全部 5 個 fold 上量測完成**（`notebooks/calibrate_folds.py`，
不重新訓練，每個 fold 用自己的驗證集重新配溫度 T，不套用其他 fold 的值）：

| fold | baseline | TTA | temperature | TTA+temperature | T（各自配的） |
|---|---|---|---|---|---|
| 0 | 1.06839 | 1.06039 | 1.06382 | 1.05904 | 1.192 |
| 1 | 1.05459 | 1.04339 | 1.05137 | 1.04256 | 1.055 |
| 2 | 1.07392 | 1.07244 | 1.07099 | 1.07019 | 1.331 |
| 3 | 1.08028 | 1.07928 | 1.07406 | 1.07358 | 1.514 |
| 4 | 1.05712 | 1.04740 | 1.05417 | 1.04711 | 1.097 |

**5/5 個 fold 都有改善**，平均改善 +0.00836（標準差 0.00288，明顯大於雜訊）——不是
單一切分的巧合，但 5 個 fold 共用同一套模型/recipe/資料分布，不是 5 個真正獨立的
實驗，不保證效果量完全轉移到隱藏測試集。T 值跨 fold 差異不小（1.055~1.514，平均
1.238，標準差 0.167），`infer_deberta.py` 套用跨 fold 平均 **T=1.238**（不是 fold 0
自己配出來的 1.192）——單一 fold 配出來的 T 對那個 fold 最準，但雜訊也最大，跨
fold 平均更穩。

**fold 0 在這次順便被重新訓練過一次**（詳見下面的 5-fold 訓練段落），驗證分數從
milestone 2 送出 143/212 那次的 1.08494 降到 1.06839——這是目前推論版本相對上次
排行榜結果的主要改善來源，TTA/校準（+0.00935）是疊加上去的第二層。推論 kernel
（`predict_logits_with_tta()` + `apply_temperature(T=1.238)`）已經用新 fold 0 權重
在 Kaggle 上跑通、格式驗證通過，並已送出排行榜：**public score 1.05159**（milestone 2
單一 fold、無 TTA/校準那次是 1.07748，改善 +0.02589）。

新增的 `llmcls/tta.py`（a/b 欄位對齊 + 平均）、`llmcls/calibration.py`
（temperature scaling）都是純 numpy，24+9 項本機測試涵蓋（不用真的模型也測得到
欄位對齊有沒有搞反、溫度配出來的方向對不對）。順手把 `PreferenceDataset` 的
tokenize 從逐列 `tokenizer.encode()` 改成整批呼叫（fast tokenizer 的平行化在批次
呼叫內部做，逐列呼叫的 FFI 開銷在 TTA 兩種順序各 tokenize 一次時尤其浪費）；
推論本身已經只需要幾分鐘，fp16/autocast + 加大 batch size 這類推論加速沒有必要，
效能心力留給訓練端（`group_by_length` 之類）。

**5 folds 全部練完**（`train_deberta.py`，驗證 milestone 3 手法是不是只在 fold 0
這份切分上剛好有效，而不是普遍成立）：

| fold | valid log loss |
|---|---|
| 0 | 1.06840 |
| 1 | 1.05461 |
| 2 | 1.07391 |
| 3 | 1.08028 |
| 4 | 1.05712 |

全部優於基準 1.09861。第一次嘗試 5 folds 一次跑完時踩到一個新坑：`train_fold()`
的 `output_dir` 同時是 `TrainingArguments` 的 checkpoint 目錄跟最終存檔目錄，
Trainer 自己 `save_steps` 存的 `checkpoint-*/`（含 optimizer/scheduler state，
體積是模型本身的 2-3 倍）訓練完沒清掉，5 個 fold 疊起來直接把 Kaggle 磁碟塞爆
（`OSError: No space left on device`，練到 fold 4 過半才炸，fold 0-3 其實都順利
存好了）。修法：`trainer.save_model()` 之後自動清掉 `output_dir` 底下的
`checkpoint-*/`，每個 fold 只留最終權重（~700MB，不是 ~2.8GB）。搶救 fold 0-3
的做法：把它們的權重下載下來、包成一個新的 Kaggle Dataset（`dataset_sources`
掛進訓練 kernel），`train_deberta.py` 開頭會先檢查有沒有掛之前的權重、有的話
複製過來跳過重新訓練——第二次只花約 1.9 小時（重練 fold 4 + 複製 fold 0-3）
就補完整個 5-fold，不用整個重來一次 8.5 小時。

**5-fold 機率平均 ensemble 已送出排行榜**：`infer_deberta.py`（每個 fold 各自
TTA + 各自的校準溫度，5 組機率取平均）push 上 Kaggle 跑通、送出排行榜：
**public score 1.04529**（上一版單 fold + TTA + 校準是 1.05159，改善 +0.0063）。
5 個 fold 的機率平均確實比單一 fold 更好，不是雜訊。

**Label smoothing 控制實驗結果：幾乎沒差，不值得重練**：第一次在 fold 0 實測
`label_smoothing=0.1`，valid log loss 1.08815，比基準 1.06840 變差 0.01975，
判定「有害」——但那次的基準跟實驗是不同時間跑的兩次訓練，之後發現分類頭初始值
沒被 seed 固定住（見下面），差距有可能大半是隨機運氣，不是 label smoothing 真的
有害。用修好種子的版本重新做了一次同一個 kernel 執行內的控制 A/B（baseline 跟
`label_smoothing=0.1` 分類頭起始值保證一致）：baseline **1.08190**、
`label_smoothing=0.1` **1.08140**，**差異只有 +0.00049**——比 TTA/校準實驗量到
的雜訊量級（標準差 0.00288）還小，代表這個差距本身就是雜訊，label smoothing
既沒有像第一次測的那麼有害，也沒有真的變好。**結論：放棄**——不是因為它有害，
是因為效果在雜訊範圍內，不值得為了看不出來的差距重練全部 5 個 fold。（順帶：
這次控制實驗的 baseline 1.08190 跟正式 5-fold 訓練 fold 0 的 1.06840 差了
0.0135，再次印證分類頭初始值真的能讓同一個 fold 飄動這個量級。）`train_fold()`
的 `label_smoothing` 參數留著（預設 0.0，不影響現有行為）。

**訓練效能評估**：`attn_implementation="sdpa"` 對 DeBERTa-v3 不支援，實測直接
報錯（`DebertaV2ForSequenceClassification does not support ...`，對應 HF issue
#28005，官方還沒補），放棄。`torch.profiler` 量過 15 步的時間分佈：耗最多 CUDA
時間的是 `aten::bmm`／`aten::linear`／`aten::gather_backward`／`aten::scatter_add_`
這類模型計算本身的運算（後兩者是 DeBERTa 相對位置編碼反向傳播），不是資料載入或
tokenize，`num_workers`/`pin_memory` 這類 dataloader 調整不會有明顯幫助。

`group_by_length=True`（把長度相近的樣本分到同一個 batch，減少 padding 浪費）
單步記憶體壓力測試（最長 16 筆組成最壞 batch）沒有 OOM（餘裕 23.6%），但**完整
fold 實測失敗**：訓練在 epoch 0.4809（原訂 2 epochs，只跑了 24%）就因為 `grad_norm`
變成 `inf` 被 `StopOnNonFiniteLoss` 攔截停止——跟這個模型已知在 fp32/peak LR 附近
容易發散是同一類問題，推測 `group_by_length` 把長句子集中到同一個 batch 讓某些
batch 的梯度震幅更劇烈，更容易踩到 fp16 數值溢位。訓練被攔停之後印出的「耗時
1256s（比基準快 79%）」「valid log loss 1.09921（比基準差 0.03081）」**兩個數字
都不是有效比較**——是訓練只跑了四分之一被緊急煞車，不是 group_by_length 真的更快
或更差。結論：**放棄**，`train_fold()` 的 `group_by_length` 參數留著（預設
False，行為不變）。

**修復 fold 4 遺失、備份完整 5-fold Dataset**：上面幾次為了做實驗 push 的精簡版
`train_kernel`（冒煙測試 + 單一實驗，拿掉主要 5-fold 訓練），每次成功跑完都會把
`train_kernel` 在 Kaggle 上的輸出整個換掉；正式使用的 fold 0-4 權重（送出 1.04529
那份）因此不再能透過 `kernel_sources` 抓回來——fold 0-3 還有安全網（獨立的
`fold0-3-checkpoints` Dataset），fold 4 完全遺失。用獨立的 `train_fold4_kernel`
（不掛 `train_kernel` 的 `kernel_sources`）補練了一次 fold 4，跟 fold 0-3 合併成
新的 `fold0-4-checkpoints` Dataset（涵蓋全部 5 folds，永久保存，不受任何 kernel
之後的 push 影響），`train_kernel/kernel-metadata.json` 的 `dataset_sources` 也
改指到這個新 Dataset。

補練 fold 4 這次意外挖出一個更根本的問題：**同一個 fold、完全相同的設定重跑一次，
valid log loss 從 1.05712 飄動到 1.07493（+0.01781），且沒有發散**。追查發現
`AutoModelForSequenceClassification.from_pretrained()`（隨機初始化新的分類頭）
是在 `Trainer` 建立、`TrainingArguments(seed=42)` 生效**之前**執行的——那個 seed
從來沒真正固定住分類頭的起始值，只固定了訓練過程（資料洗牌、dropout）的隨機性。
這代表**label smoothing 第一次判定「變差 0.01975」、group_by_length 那次比較，
都可能有一部分（甚至大部分）只是分類頭初始值不同造成的隨機波動，不是那個手法
真的有害**。`train_fold()` 已經修正（`from_pretrained()` 之前先呼叫
`set_seed()`），label smoothing 用修好的版本重新驗證過（見上面「控制實驗結果」
段落），group_by_length 的發散結論維持，但信心度較低，還沒有用修好的版本重新
驗證過。

**下一步**：milestone 3 剩下訓練時 a/b 對調增強、`winner_tie` 特殊處理還沒
開始；訓練效能這條線目前評估過的候選手法（sdpa、group_by_length、dataloader
調整）全部放棄或無效，`torch.compile()` 因為 DeBERTa 自訂運算容易觸發頻繁
重新編譯，評估認為優先度太低沒有試。

## 路線圖

- [x] 里程碑 0：pipeline 跑通 + 類別先驗 baseline
  （先驗只比 ln(3) 好一點點，本來就是如此 —— 它的用途是驗證管線，不是拿分策略）
- [x] 里程碑 1：下載真實資料，用真實 schema 重跑上面全部流程
- [x] 里程碑 2：DeBERTa-v3-base 三分類微調，fold 0 跑出 valid log loss 1.08494，
      優於基準，已送出排行榜 143/212。訓練加速後單一 fold 約 1.7 小時（原本
      將近 9 小時）。**5 folds 已全部練完**（1.06840 / 1.05461 / 1.07391 /
      1.08028 / 1.05712，全數優於基準）。**fold 4 後來因為 Kaggle 輸出被實驗
      push 蓋掉而遺失，補練一次分數變成 1.07493**（同一份設定重跑就飄動 0.0178，
      見下面 seed 漏洞的說明）——目前 `fold0-4-checkpoints` Dataset 裡的 fold 4
      是這個補練版本，跟送出 1.04529 排行榜那次用的不是同一份權重，但都優於基準。
- [ ] 里程碑 3：加上已知有效的手法
  - [x] a/b 對調 TTA + temperature scaling：5 folds 各自驗證，5/5 都有改善
        （平均 +0.00836），已套進 `infer_deberta.py`（fold 0 + T=1.238），
        已送出排行榜：public score 1.05159（milestone 2 為 1.07748）
  - [x] 5-fold 機率平均 ensemble：`infer_deberta.py` push 上 Kaggle 送排行榜，
        public score 1.04529（單 fold + TTA + 校準是 1.05159）
  - [x] label smoothing：控制好種子重測（同一次 kernel 執行內 baseline vs
        `label_smoothing=0.1`），差異只有 +0.00049（比雜訊量級 0.00288 還小）
        ——**放棄**（第一次測的「變差 0.01975」是分類頭初始值沒固定住的偽陽性）
  - [ ] a/b 對調當「訓練時」的資料增強（目前只做了推論時的 TTA，訓練資料還沒加這個增強）
  - [ ] 針對 `winner_tie` 這一類的處理（通常是最難、也是 loss 的主要來源）
  - [x] 訓練效能：`attn_implementation="sdpa"` 不支援（HF issue #28005）、
        `group_by_length=True` 完整 fold 實測在 epoch 0.4809 就發散（grad_norm
        變 inf）、dataloader 調整經 profiling 確認沒有幫助——三個候選手法**全部
        放棄**，`torch.compile()` 評估後判斷優先度太低沒有試
- [ ] 里程碑 4：換更大的模型 + LoRA + 4bit 量化（需要自有 GPU 或雲端 GPU）

## 參考

- 母體賽事 [LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
  有完整的前段班解法可讀，是這題最有價值的參考資料。
