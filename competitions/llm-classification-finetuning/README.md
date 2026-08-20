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
  1.08575**（基準 1.09861，改善 +0.01286，`n_nonfinite: 0`，全程無發散）。
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
`submission.csv`。里程碑 2 訂的「走完一次 Kaggle Notebook 提交」的流程部分已完成
——實際送出到排行榜（Kaggle 網頁的 Submit 按鈕）還沒做，等你確認要不要送。

## 路線圖

- [x] 里程碑 0：pipeline 跑通 + 類別先驗 baseline
  （先驗只比 ln(3) 好一點點，本來就是如此 —— 它的用途是驗證管線，不是拿分策略）
- [x] 里程碑 1：下載真實資料，用真實 schema 重跑上面全部流程
- [x] 里程碑 2：DeBERTa-v3-base 三分類微調，fold 0 跑出 valid log loss 1.08575，
      優於基準。下一步：跑滿 5 folds、把推論 notebook 走完一次 submission.csv。
- [ ] 里程碑 3：加上已知有效的手法
  - a/b 對調做資料增強，推論時對兩種順序各跑一次再平均（TTA）—— 對付位置偏誤
  - label smoothing / temperature scaling / 事後校準 —— log loss 吃的是機率校準，不是準確率
  - 針對 `winner_tie` 這一類的處理（通常是最難、也是 loss 的主要來源）
- [ ] 里程碑 4：換更大的模型 + LoRA + 4bit 量化（需要自有 GPU 或雲端 GPU）

## 參考

- 母體賽事 [LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
  有完整的前段班解法可讀，是這題最有價值的參考資料。
