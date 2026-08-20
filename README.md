# Kaggle 工作區

一個 repo 裝多場競賽。每場競賽是 `competitions/` 底下一個自足的目錄，
目錄名就是 Kaggle 的競賽 slug（網址 `kaggle.com/competitions/<slug>` 的那一段）。

競賽之間**不共用程式碼**：資料 schema、評分指標、模型都不一樣，硬抽共用層只會綁死彼此。
真的出現重複到值得抽的東西時再抽，現在不預先設計。

## 競賽

| 競賽 | 指標 | 狀態 |
| --- | --- | --- |
| [llm-classification-finetuning](competitions/llm-classification-finetuning/) | multi-class log loss（亂猜 = ln(3) ≈ 1.0986） | 里程碑 2 完成：DeBERTa-v3-base fold 0 valid log loss 1.08575，優於基準。下一步：5 folds + 提交 |

## 目錄慣例

```
competitions/<kaggle-slug>/
  README.md              該競賽的題目說明、狀態、路線圖 —— 進競賽前先讀這個
  src/<pkg>/             可 import 的模組（本機與 Kaggle Notebook 共用同一份）
  scripts/               CLI 入口：下載資料、產 fixture、跑 baseline
  tests/                 本機單元測試
  notebooks/             Kaggle Notebook 草稿
  data/                  競賽資料（git 忽略）
  outputs/               提交檔與產出（git 忽略）
  requirements.txt       該競賽的相依（各競賽版本可能互相衝突，所以不共用）
```

競賽目錄內的程式碼一律用 `Path(__file__).resolve().parents[n]` 推導自己的競賽根目錄，
不依賴 cwd 也不依賴 git repo 根目錄。**指令則要在該競賽目錄內執行**，
因為 `--data-dir data/fixture` 這類參數是相對於 cwd 解析的。

## Python 環境

共用一個放在工作區根目錄的 `.venv`（venv 內部寫死絕對路徑、搬不動，所以固定在這）：

```bash
uv venv                       # 在工作區根目錄；已存在的話會被重建掉，只在首次跑
cd competitions/<slug>
uv pip install --python ../../.venv/bin/python -r requirements.txt
../../.venv/bin/python -m pytest tests/ -q
```

哪天某場競賽的相依跟別場打架，就在該競賽目錄自己建一個 `.venv` —— `.gitignore`
的 `.venv/` 是任意層級都蓋得到的，不用改設定。

## 新增一場競賽

```bash
mkdir -p competitions/<kaggle-slug>/{src,scripts,tests,notebooks,data,outputs}
touch competitions/<kaggle-slug>/{data,outputs}/.gitkeep
```

`.gitkeep` 要記得建：`.gitignore` 忽略 `**/data/*` 與 `**/outputs/*`，
只放行這兩個檔，沒有它們的話空目錄不會進 git。

## git 忽略規則

`.gitignore` 只有工作區根目錄這一份，涵蓋所有競賽。含斜線的 pattern 錨定在
`.gitignore` 所在目錄，所以資料與產出的規則都寫成 `**/data/*`、`**/outputs/*` —— 
少了 `**` 就只蓋得到根目錄，競賽子目錄裡的競賽資料會變成可追蹤（**競賽規則禁止散布資料**）。

改完規則用這個驗證實際命中哪一條：

```bash
git check-ignore -v competitions/<slug>/data/some_file.json
```
