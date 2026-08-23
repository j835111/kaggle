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

## `train_kernel/kernel-metadata.json` 的 `dataset_sources` 是磁碟爆掉那次的急救殘留

`"dataset_sources": ["jameslin45/llm-classification-fold0-3-checkpoints"]` 是
disk 塞爆事故的搶救設定——那次 fold 0-3 其實都練完了、只有 fold 4 沒存到，把
fold 0-3 的權重包成 Dataset 掛回去，`train_deberta.py` 開頭的複製邏輯
（`pathlib.Path("/kaggle/input").glob("**/fold*__model.safetensors")`，找到就複製、
`MODEL_DIR/fold{N}` 已存在就跳過訓練）看到就直接抄過去、不會再练一次。

這個掛載到現在還留著，**沒清掉**。後果：只要 push 這個訓練 kernel，fold 0-3 永遠
被那個舊 Dataset 檔住，主迴圈永遠只會真的訓練不在那個舊 Dataset 裡的 fold（目前
是 fold 4）。想要「這次 push 真的把 5 個 fold 全部用新設定重練一次」（例如
label smoothing 驗證有效、要正式套用時），**先把這個 `dataset_sources` 清空**，
不然新設定只會套用到 fold 4，fold 0-3 還是舊的沒套用新設定的權重。

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
