# llm-classification-finetuning

專案背景、目前進度、路線圖見 `README.md`。這份檔案只放「怎麼在這個專案裡做事」的
操作結論，不重複 README 已經有的內容。

## Label smoothing / 訓練時資料增強這類手法，改了就得整批重練

`llmcls/tta.py`（TTA）、`llmcls/calibration.py`（temperature scaling）、5-fold
機率平均 ensemble 都是**推論階段**的手法：模型權重不變，只是推論時多做一點事，
所以可以直接套在已經練好的權重上，不用重新訓練。

`label_smoothing`（`train_fold()` 的參數）跟訓練時的 a/b 對調增強不一樣——這兩個
是**改訓練過程本身**（loss 怎麼算、看到什麼訓練資料），已經練好的舊權重是在
「沒有這個改動」的情況下練出來的，沒辦法事後補，只能整批重新訓練。決定要套用
這類手法之前，先在 fold 0 花一次訓練時間單獨驗證有沒有用（跟主要的 5-fold 訓練
分開存，不要覆蓋掉正在用的權重），確認贏過雜訊量級（TTA/校準那次量到的 fold 間
標準差是 0.00288）才值得整批重練。

## 正式權重只信 Dataset，不要信任何 kernel 的「最新輸出」

`train_kernel` 這幾次為了做實驗 push 的精簡版 notebook（冒煙測試 + 單一實驗，
拿掉主要 5-fold 訓練），每次成功跑完都會把 `train_kernel` 在 Kaggle 上「最新一次
的輸出」整個換掉。第一次踩到這個坑：正式使用的 fold 0-4 權重（送出 1.04529 排行榜
那份）因此不見了，fold 4 完全遺失、只能重練（見 README「修復 fold 4 遺失」段落）。

**現在的做法**：`jameslin45/llm-classification-fold0-4-checkpoints` 是涵蓋全部
5 folds 的**獨立 Dataset**（不是任何 kernel 的輸出，不會被 push 影響），
`train_kernel/kernel-metadata.json` 的 `dataset_sources` 指到這個 Dataset——
`train_deberta.py` 開頭的複製邏輯（`pathlib.Path("/kaggle/input").glob(
"**/fold*__model.safetensors")`，找到就複製、`MODEL_DIR/fold{N}` 已存在就跳過
訓練）看到 5 個 fold 都在，會全部跳過訓練。想要「這次 push 真的把 5 個 fold
全部用新設定重練一次」（例如某個訓練時的手法驗證有效、要正式套用時），**先把
這個 `dataset_sources` 清空**，重練完務必比照這次的做法，把輸出打包成新的
Dataset 更新掉舊的，不要只依賴 kernel 的輸出。

`infer_kernel`/`calibrate_kernel` 目前還是用 `kernel_sources` 接 `train_kernel`
的輸出，不是直接接這個 Dataset——只要 `train_kernel` 之後的 push 都是完整跑主要
5-fold 迴圈（不是精簡版實驗），迴圈會全部從 Dataset 複製回來，`train_kernel` 的
輸出仍然正確，鏈路沒斷。但如果又像這次一樣 push 精簡版實驗、跳過主迴圈，
`infer_kernel`/`calibrate_kernel` 抓到的又會是不完整的權重——這是還沒補的
根本修法（改成直接 `dataset_sources` 接 Dataset，不透過 `kernel_sources`）。

## `train_fold()` 的分類頭初始化，`TrainingArguments(seed=42)` 固定不住

`AutoModelForSequenceClassification.from_pretrained()`（隨機初始化新的分類頭）
是在 `Trainer` 建立、`TrainingArguments(seed=42)` 生效**之前**執行的——那個 seed
只固定得了訓練過程本身（資料洗牌順序、dropout），固定不了分類頭的起始隨機值。
實測同一個 fold、完全相同設定重跑一次，valid log loss 可以飄動 0.01~0.02，這個
量級足以讓一次性的 A/B 比較（沒有控制初始化）得出錯誤結論——label smoothing
第一次判定「有害」就是在這個漏洞修好之前測的，後來才發現差距有一部分可能只是
隨機運氣。`train_fold()` 現在會在 `from_pretrained()` 之前呼叫 `set_seed(SEED)`，
之後任何新的單次訓練實驗（不是像 TTA/校準那種直接對已訓練權重做的事）都要在
**同一次 kernel 執行裡**同時跑 baseline 跟要測的版本，不要拿新跑的結果去跟舊
session 的歷史數字比較。

**已用這個修法重測過 label smoothing，結果印證了這個漏洞的嚴重性**：同一次
kernel 執行內控制好種子的 baseline 是 1.08190，跟正式 5-fold 訓練 fold 0 的
1.06840 差了 0.0135——同一個 fold、同一組超參數，只因為分類頭起始值不同就飄動
這個量級，證實漏洞修好前的比較（label smoothing、group_by_length）confidence
確實該打折扣。控制好之後，baseline 1.08190 vs `label_smoothing=0.1` 1.08140，
差異只有 +0.00049（比 TTA/校準量到的雜訊量級 0.00288 還小）——第一次判定的
「變差 0.01975」是偽陽性，真實結論是「幾乎沒差，不值得重練」，不是「有害」。

## Kaggle CLI（2.2.4）沒有「停止正在跑的 kernel」這個指令

`kaggle kernels --help` 列出來的只有
`list/files/get/init/push/pull/output/status/logs/update/delete/topics`。
`push` 之後如果發現這次跑的東西有一段是白工（例如上面那個 fold 0-3 一直被跳過的
問題），沒有辦法中途喊停、只留已經跑完的那部分結果——`delete` 是把整個 kernel
刪掉，不是「停止目前這次執行」。想避免白花 GPU 時間，只能在 push 之前先想清楚
這次要跑的東西有沒有多餘的步驟，不要事後補救。

## 本機沒有 torch/transformers

`src/llmcls/train.py` 只能在 Kaggle Notebook 或有 GPU 的機器 import，本機
（共用 venv `../../.venv/`）沒裝 torch/transformers，改動這個檔案不能用
`pytest tests/` 驗證邏輯對不對，只能靠 review + 實際 push 上 Kaggle 跑。
`notebooks/*.py`（train_deberta.py / infer_deberta.py / calibrate_folds.py）
改動後要記得 `python scripts/build_notebooks.py` 重新產生對應的 `.ipynb`
才會反映到實際 push 上去的檔案。
