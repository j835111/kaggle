#!/usr/bin/env bash
# 下載競賽資料到 data/。
#
# 先決條件（兩者缺一都會失敗）：
#   1. 到 https://www.kaggle.com/competitions/llm-classification-finetuning/rules
#      按下 "I Understand and Accept" 接受競賽規則。
#   2. 認證擇一：
#      - 舊版：從 https://www.kaggle.com/settings 產生 API token，
#        放到 ~/.kaggle/kaggle.json 並 chmod 600。
#      - 新版（kaggle CLI 2.x）：執行 `kaggle login`（瀏覽器登入），
#        token 存在 ~/.kaggle/access_token。
#
# 用法：bash scripts/download_data.sh
set -euo pipefail

COMPETITION="llm-classification-finetuning"
COMP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${COMP_ROOT}/data"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "找不到 kaggle CLI。請先安裝：uv pip install kaggle（或 pip install kaggle）" >&2
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" && ! -f "${HOME}/.kaggle/access_token" ]]; then
  echo "找不到 ${HOME}/.kaggle/kaggle.json 或 ${HOME}/.kaggle/access_token —— 請先完成 Kaggle 認證。" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}"
kaggle competitions download -c "${COMPETITION}" -p "${DATA_DIR}"
unzip -o "${DATA_DIR}/${COMPETITION}.zip" -d "${DATA_DIR}"
rm -f "${DATA_DIR}/${COMPETITION}.zip"

echo "完成。data/ 內容："
ls -la "${DATA_DIR}"
