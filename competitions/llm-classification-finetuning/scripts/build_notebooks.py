#!/usr/bin/env python3
"""把 notebooks/*.py（Jupytext percent format 草稿）轉成可直接上傳 Kaggle 的 .ipynb。

順便把草稿裡的 bootstrap 佔位 cell（`# >>> 這裡貼 ... <<<`）換成
scripts/gen_notebook_bootstrap.py 產生的真正原始碼 —— 產出的 .ipynb 完全自足，
不需要使用者手動貼任何東西，也不需要另外建 Kaggle Dataset 掛程式碼。

用法：

    python scripts/build_notebooks.py

會讀 notebooks/train_deberta.py、notebooks/infer_deberta.py，寫出對應的 .ipynb
到 notebooks/train_kernel/train_deberta.ipynb、notebooks/infer_kernel/infer_deberta.ipynb
（跟各自的 kernel-metadata.json 放在同一個資料夾，方便直接 `kaggle kernels push -p`）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

COMP_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = COMP_ROOT / "notebooks"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_notebook_bootstrap import build_bootstrap_source  # noqa: E402

BOOTSTRAP_PLACEHOLDER = "# >>> 這裡貼 notebooks/_bootstrap_cell.py 的完整內容 <<<"
CELL_MARKER = re.compile(r"^# %%(\s*\[markdown\])?\s*$")


def parse_percent_cells(text: str) -> list[tuple[str, str]]:
    """回傳 [(cell_type, source), ...]。cell_type 是 'markdown' 或 'code'。"""
    lines = text.splitlines()
    cells: list[tuple[str, list[str]]] = []
    current_type: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_type is not None:
            # 去掉 cell 前後多餘的空行（cell 之間的空行是分隔符，不是內容）。
            body = current_lines[:]
            while body and body[0] == "":
                body.pop(0)
            while body and body[-1] == "":
                body.pop()
            cells.append((current_type, body))

    for line in lines:
        m = CELL_MARKER.match(line)
        if m:
            flush()
            current_type = "markdown" if m.group(1) else "code"
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    out: list[tuple[str, str]] = []
    for cell_type, body_lines in cells:
        if cell_type == "markdown":
            stripped = [ln[2:] if ln.startswith("# ") else ln[1:] if ln == "#" else ln for ln in body_lines]
            out.append(("markdown", "\n".join(stripped)))
        else:
            # 註解掉的 `# !pip install ...` 之類的行保持原樣、不自動變成會執行的 magic ——
            # 這些預設是「不需要才留著、要用再手動打開」，自動解開等於每次都強制升級套件。
            source = "\n".join(body_lines)
            if source.strip() == BOOTSTRAP_PLACEHOLDER:
                source = build_bootstrap_source().rstrip("\n")
            out.append(("code", source))
    return out


def to_notebook_json(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for cell_type, source in cells:
        src_lines = source.splitlines(keepends=True)
        cell = {"cell_type": cell_type, "metadata": {}, "source": src_lines}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_one(py_path: Path, out_path: Path) -> None:
    cells = parse_percent_cells(py_path.read_text(encoding="utf-8"))
    nb = to_notebook_json(cells)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{py_path.relative_to(COMP_ROOT)} -> {out_path.relative_to(COMP_ROOT)}  ({len(cells)} cells)")


def main() -> int:
    build_one(NOTEBOOKS_DIR / "train_deberta.py", NOTEBOOKS_DIR / "train_kernel" / "train_deberta.ipynb")
    build_one(NOTEBOOKS_DIR / "infer_deberta.py", NOTEBOOKS_DIR / "infer_kernel" / "infer_deberta.ipynb")
    build_one(NOTEBOOKS_DIR / "calibrate_folds.py", NOTEBOOKS_DIR / "calibrate_kernel" / "calibrate_folds.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
