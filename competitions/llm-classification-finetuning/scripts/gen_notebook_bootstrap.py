#!/usr/bin/env python3
"""把 src/llmcls/*.py 打包成一個「貼進 Kaggle Notebook 第一個 cell」的 bootstrap 腳本。

背景：Kaggle Notebook 沒辦法直接 import 本機的 src/llmcls，得先把它送進 Kaggle 環境。
掛 Kaggle Dataset 是一種做法，但容易在「有沒有建對 Dataset / 掛載路徑對不對」上出錯
（實測就真的出過 ModuleNotFoundError）。這裡改用更可靠的方式：把每個檔案的原始碼
內嵌成字串，貼進 notebook 執行後直接在 /kaggle/working 寫出同樣的檔案結構、
把它加進 sys.path —— 不需要另外上傳、掛載任何東西。

用法：

    python scripts/gen_notebook_bootstrap.py > notebooks/_bootstrap_cell.py

輸出的檔案內容就是可以直接貼進 Kaggle Notebook 第一個 cell 的程式碼。
src/llmcls/ 有任何改動後要重新產生一次，否則 Kaggle 上跑的是舊版程式碼。
"""

from __future__ import annotations

import sys
from pathlib import Path

COMP_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = COMP_ROOT / "src" / "llmcls"

# 生成時用 r''' 當 Python 字串分隔符（raw string——原始碼裡一堆 "\n"、r"\s+" 這類跳脫
# 序列，用非 raw 字串包起來會被 bootstrap cell 自己的解析器吃掉，寫出來的檔案內容會被
# 悄悄改掉）。前提是原始碼裡不會出現 '''（本專案的 docstring 一律用 """）且不會以單一
# 反斜線結尾（raw string 語法限制）。這裡先驗證一次，壞掉要當場報錯，而不是產生一份
# 貼進 Kaggle 會 SyntaxError、或更糟——悄悄跑錯程式碼——的檔案。
DELIM = "'''"


def build_bootstrap_source() -> str:
    """回傳可以直接貼進 Kaggle Notebook 一個 cell 的完整原始碼字串。

    也被 scripts/build_notebooks.py 呼叫，內嵌進 .ipynb 的第一個 code cell，
    所以邏輯只寫這一份，CLI 用法（印到 stdout）跟 .ipynb 產生器共用。
    """
    files = sorted(SRC_DIR.glob("*.py"))
    if not files:
        raise FileNotFoundError(f"找不到任何檔案：{SRC_DIR}")

    for f in files:
        content = f.read_text(encoding="utf-8")
        if DELIM in content:
            raise ValueError(f"{f} 含有 {DELIM}，無法用這個分隔符內嵌，請改寫 gen_notebook_bootstrap.py")
        if content.endswith("\\"):
            raise ValueError(f"{f} 內容以反斜線結尾，raw string 包不住，請改寫 gen_notebook_bootstrap.py")

    lines = [
        '"""貼進 Kaggle Notebook 第一個 cell —— 由 scripts/gen_notebook_bootstrap.py 產生，不要手改。"""',
        "",
        "import pathlib",
        "import sys",
        "",
        "_llmcls_files = {",
    ]
    for f in files:
        content = f.read_text(encoding="utf-8")
        lines.append(f"    {f.name!r}: r{DELIM}{content}{DELIM},")
    lines += [
        "}",
        "",
        # 刻意複製本機 <comp_root>/src/llmcls/ 這個層級結構（不是省一層扁平的
        # <base>/llmcls/），因為 llmcls/config.py 用 Path(__file__).resolve().parents[2]
        # 往上兩層猜 comp root；層數對不上，DATA_DIR 猜錯路徑的 fallback 分支就會找錯地方
        # （實測就真的因為這樣 fallback 到 /kaggle/working/data 而找不到 train.csv）。
        '_pkg_dir = pathlib.Path("/kaggle/working/llmcls_src/src/llmcls")',
        "_pkg_dir.mkdir(parents=True, exist_ok=True)",
        "for _name, _content in _llmcls_files.items():",
        '    (_pkg_dir / _name).write_text(_content, encoding="utf-8")',
        "",
        'sys.path.insert(0, "/kaggle/working/llmcls_src/src")',
        'print("llmcls bootstrapped:", sorted(p.name for p in _pkg_dir.glob("*.py")))',
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        print(build_bootstrap_source(), end="")
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
