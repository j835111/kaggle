#!/usr/bin/env python3
"""里程碑 0：類別先驗（class prior）baseline。

不看任何文字內容，直接輸出訓練集的類別比例當作預測。這是唯一保證優於
均勻亂猜 ln(3) 的零成本做法，用途是**驗證整條 pipeline 通了**：
資料解析 → group CV → log loss → 提交檔組裝與格式檢查。

任何真正的模型都必須贏過這裡印出的分數，否則等於沒學到東西。

    python scripts/baseline_prior.py                      # 用真實資料 (data/)
    python scripts/baseline_prior.py --data-dir data/fixture   # 用合成 fixture 煙霧測試
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.config import LABEL_COLS, N_CLASSES, N_FOLDS  # noqa: E402
from llmcls.cv import add_folds, fold_indices, prompt_group_key  # noqa: E402
from llmcls.data import load_test, load_train  # noqa: E402
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss  # noqa: E402
from llmcls.submission import build_submission, save_submission  # noqa: E402


def class_prior(labels: np.ndarray) -> np.ndarray:
    """帶 Laplace 平滑的類別比例，避免某類在 fold 內為 0 造成 log loss 爆炸。"""
    counts = np.bincount(labels, minlength=N_CLASSES) + 1.0
    return counts / counts.sum()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="覆寫資料目錄")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--out", default="submission_prior.csv")
    args = parser.parse_args()

    data_dir = args.data_dir
    train_csv = data_dir / "train.csv" if data_dir else None
    test_csv = data_dir / "test.csv" if data_dir else None
    sample_csv = data_dir / "sample_submission.csv" if data_dir else None

    train = load_train(train_csv)
    print(f"train: {len(train)} 列")

    train = add_folds(train, n_folds=args.folds)
    n_groups = prompt_group_key(train).nunique()
    print(f"以 prompt 分組切成 {args.folds} folds（{n_groups} 個唯一 prompt，無跨 fold 洩漏）")

    # ---- CV ----
    oof = np.zeros((len(train), N_CLASSES))
    labels = train["label"].to_numpy()
    for fold in range(args.folds):
        tr_idx, va_idx = fold_indices(train, fold)
        prior = class_prior(labels[tr_idx])
        oof[va_idx] = prior
        print(f"  fold {fold}: valid {len(va_idx):>6} 列  log loss {log_loss(labels[va_idx], oof[va_idx]):.5f}")

    cv_score = log_loss(labels, oof)
    delta = UNIFORM_LOGLOSS - cv_score
    print(f"\nOOF log loss     {cv_score:.5f}")
    print(f"均勻亂猜基準      {UNIFORM_LOGLOSS:.5f}  (ln 3)")
    print(f"改善              {delta:+.5f}  {'✓ 優於基準' if delta > 0 else '✗ 未優於基準'}")
    print("整體類別比例      " + ", ".join(f"{c}={p:.3f}" for c, p in zip(LABEL_COLS, class_prior(labels))))

    # ---- 產生提交檔 ----
    test = load_test(test_csv)
    probs = np.tile(class_prior(labels), (len(test), 1))
    sub = build_submission(test["id"], probs)
    path = save_submission(sub, name=args.out, sample_path=sample_csv)
    print(f"\n提交檔已寫入並通過格式檢查：{path}  ({len(sub)} 列)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
