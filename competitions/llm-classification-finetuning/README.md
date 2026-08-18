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

- **任何牽涉 torch / transformers 的程式碼**（`llmcls/train.py`、`scripts/train.py`、
  兩份 Kaggle Notebook 草稿）：本機無 GPU，程式碼已寫但完全沒在真的環境跑過。
  第一次在 Kaggle Notebook 上跑（`train_deberta.py`）就是這批程式碼的第一次實測。

## 路線圖

- [x] 里程碑 0：pipeline 跑通 + 類別先驗 baseline
  （先驗只比 ln(3) 好一點點，本來就是如此 —— 它的用途是驗證管線，不是拿分策略）
- [x] 里程碑 1：下載真實資料，用真實 schema 重跑上面全部流程
- [ ] 里程碑 2：DeBERTa-v3-base 三分類微調，prompt + 兩份回覆串接、head+tail 截斷。
      目標是做出**第一個明顯優於 1.0986** 的分數，並走完一次 Kaggle Notebook 提交
      **程式碼已就緒**（`llmcls/train.py` + `notebooks/train_deberta.py` /
      `infer_deberta.py`），還沒在 Kaggle 上實跑過。下一步：先跑 fold 0、2 epochs，
      確認 valid log loss < 1.0986 再考慮跑滿 5 folds。
- [ ] 里程碑 3：加上已知有效的手法
  - a/b 對調做資料增強，推論時對兩種順序各跑一次再平均（TTA）—— 對付位置偏誤
  - label smoothing / temperature scaling / 事後校準 —— log loss 吃的是機率校準，不是準確率
  - 針對 `winner_tie` 這一類的處理（通常是最難、也是 loss 的主要來源）
- [ ] 里程碑 4：換更大的模型 + LoRA + 4bit 量化（需要自有 GPU 或雲端 GPU）

## 參考

- 母體賽事 [LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
  有完整的前段班解法可讀，是這題最有價值的參考資料。
