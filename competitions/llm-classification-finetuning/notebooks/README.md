# Kaggle Notebook 草稿

這是 code competition，推論必須在 Kaggle Notebook 內完成，且**正式評分時網路會關閉**、
`test.csv` 會換成真正的隱藏測試集。所以訓練跟推論拆成兩個 notebook：

- `train_deberta.py` —— 訓練。開網路（下載 base model）+ GPU，練完把權重存到
  `/kaggle/working/model/fold0/`。
- `infer_deberta.py` —— 推論。**關網路**，讀訓練 notebook 存出的權重，對（真正的）
  `test.csv` 跑推論，寫出 `submission.csv`。

這兩個 `.py` 是 `# %%` cell 分隔的草稿（Jupytext percent format），是**編輯用的原始
檔**，不要直接上傳。實際上傳 Kaggle 的是 `train_kernel/train_deberta.ipynb` 與
`infer_kernel/infer_deberta.ipynb`，由 `scripts/build_notebooks.py` 從草稿產生。

## 用 CLI 直接推上 Kaggle（推薦）

```bash
# 每次改了 notebooks/*.py 或 src/llmcls/ 之後都要重新產生一次：
../../.venv/bin/python scripts/build_notebooks.py

kaggle kernels push -p notebooks/train_kernel   # 上傳並立刻觸發執行
kaggle kernels status jameslin45/llm-classification-train-deberta-fold-0
```

`train_kernel/kernel-metadata.json` 已經設好 GPU、Internet On、`competition_sources`
掛這場競賽的資料，`push` 之後 Kaggle 就會直接開始跑，不需要再手動到網頁設定。

**注意**：Kaggle 的 kernel push 實測會忽略 `kernel-metadata.json` 裡手動取的 `id`
slug，改用 `title` 轉出來的 slug（例如 title 是 `LLM Classification - Train DeBERTa
(fold 0)`，實際 slug 變成 `llm-classification-train-deberta-fold-0`，不是我們原本
取的 `llmcls-train-deberta`）。push 完務必用 `kaggle kernels status` 或網頁 URL
確認實際 slug，並把 `kernel-metadata.json` 的 `id` 欄位改成一致，下次 push 才會更新
同一個 kernel而不是又生出一個新的。

訓練 kernel 確認 valid log loss 贏過 1.0986 之後，再推推論 kernel ——
`infer_kernel/kernel-metadata.json` 用 `kernel_sources` 接了訓練 kernel 的輸出
（`jameslin45/llm-classification-train-deberta-fold-0`）。**掛載路徑不要用猜的**：
實測 `kernel_sources` 的掛載路徑跟 `competition_sources` 一樣，跟直覺猜的不一樣
（第一次猜 `/kaggle/input/<kernel-slug>/model/fold0`，實際掛載位置不同，直接
`HFValidationError` 找不到），`infer_deberta.py` 現在改成用
`pathlib.Path("/kaggle/input").glob("**/fold0/model.safetensors")` 動態找出真正
的路徑，不管 Kaggle 這次掛在哪都能動：

```bash
kaggle kernels push -p notebooks/infer_kernel
kaggle kernels status jameslin45/llm-classification-infer-deberta-submission
kaggle kernels output jameslin45/llm-classification-infer-deberta-submission -p outputs/infer_kernel_output
```

訓練跟推論之間還有一個**校準實驗** kernel（`calibrate_folds.py` /
`calibrate_kernel/`）——不重新訓練，對訓練 kernel 存出的每個 fold 各自在自己的
驗證集上量測 TTA（a/b 對調平均）、temperature scaling（每個 fold 各自重新配 T，
不套用其他 fold 配出來的值）、兩者疊加分別能不能贏過 baseline，5 個 fold 一起看
才知道這個改善是不是單一切分的巧合，決定哪個組合值得套進 `infer_deberta.py`：

```bash
kaggle kernels push -p notebooks/calibrate_kernel
kaggle kernels status jameslin45/llm-classification-calibrate-deberta-all-folds
kaggle kernels output jameslin45/llm-classification-calibrate-deberta-all-folds -p outputs/calibrate_kernel_output
```

`kernel_sources` 跟 `infer_kernel` 一樣接訓練 kernel 的輸出、`enable_internet: false`
——只是讀本地權重量測分數，不用連網路。

## 手動貼到 Kaggle 網頁（備用）

不想用 CLI 的話，把 `train_deberta.py` / `infer_deberta.py` 每個 `# %%` 區塊貼成
Kaggle Notebook 的一個 cell。第一個 code cell 是
`>>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<` 這行佔位文字，要換成
`notebooks/_bootstrap_cell.py` 的完整內容（`python scripts/gen_notebook_bootstrap.py
> notebooks/_bootstrap_cell.py` 產生，`src/llmcls/` 有改動要重新產生）。執行後會把
原始碼直接寫進 `/kaggle/working/llmcls_src/` 並加進 `sys.path`，`import llmcls` 就
能用 —— 不需要另外建 Kaggle Dataset 掛程式碼，這條路徑之前在「Dataset 有沒有建對 /
掛載名稱對不對」上出過 `ModuleNotFoundError`。

這條路徑下 Settings 要自己設（Internet On/Off、Add Data）；`LLMCLS_MODEL_DIR` 不用
自己改，`infer_deberta.py` 會自動 glob 找到掛進來的 checkpoint。

## 共用的部分

- `llmcls.config` 偵測到 `/kaggle/input/` 存在時會自動切換 `DATA_DIR`；`MODEL_DIR`
  則是用 `LLMCLS_MODEL_DIR` 環境變數指到模型的實際掛載路徑。
- 提交前一定要跑 `validate_submission()`（`save_submission()` 內部已經會呼叫）；
  Kaggle 只會回報「格式錯誤」，不會告訴你錯在哪一列。
- **本機看到的 `test.csv` 只有 3 列**，是格式範例，不是真正的評分資料 —— 推論程式碼
  不能假設列數，也不能依賴看得到的那 3 列去快取或調參。
