# Kaggle Notebook 草稿

這是 code competition，推論必須在 Kaggle Notebook 內完成，且**正式評分時網路會關閉**、
`test.csv` 會換成真正的隱藏測試集。所以訓練跟推論拆成兩個 notebook：

- `train_deberta.py` —— 訓練。開網路（下載 base model）+ GPU，練完把權重存到
  `/kaggle/working/model/`，Save Version 後那個輸出資料夾會變成一個 Kaggle Dataset。
- `infer_deberta.py` —— 推論。**關網路**，掛上訓練 notebook 存出的模型 Dataset，
  對（真正的）`test.csv` 跑推論，寫出 `submission.csv`。

兩個檔案都是 `# %%` cell 分隔的草稿（Jupytext percent format），不是可直接上傳的
`.ipynb`。用法：在 Kaggle 開一個新 Notebook，把每個 `# %%` 區塊的內容貼成一個 cell
（或用 `jupytext --to notebook` 在本機先轉成 `.ipynb` 再上傳）。

共用的部分：

1. **把 `src/llmcls` 送進 Kaggle**：兩份草稿的第一個 code cell 都寫著
   `>>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<`。先在本機執行
   `python scripts/gen_notebook_bootstrap.py > notebooks/_bootstrap_cell.py`
   （`src/llmcls/` 有改動就要重新產生一次），再把產生的檔案內容整個貼進那個 cell。
   執行後會把原始碼直接寫進 `/kaggle/working/llmcls_src/` 並加進 `sys.path`，
   `import llmcls` 就能用了 —— 不需要另外建 Kaggle Dataset 掛程式碼，這條路徑
   曾經在「Dataset 有沒有建對 / 掛載名稱對不對」上出過 `ModuleNotFoundError`。
2. `llmcls.config` 偵測到 `/kaggle/input/` 存在時會自動切換 `DATA_DIR`；`MODEL_DIR`
   則是用 `LLMCLS_MODEL_DIR` 環境變數指到模型 Dataset 的掛載路徑（掛載名稱由你在
   Kaggle UI 上決定，沒辦法預先寫死 —— 訓練完在 Kaggle 的 Output 分頁確認實際路徑）。
3. 提交前一定要跑 `validate_submission()`（`save_submission()` 內部已經會呼叫）；
   Kaggle 只會回報「格式錯誤」，不會告訴你錯在哪一列。

**本機看到的 `test.csv` 只有 3 列**，是格式範例，不是真正的評分資料 —— 推論程式碼
不能假設列數，也不能依賴看得到的那 3 列去快取或調參。
