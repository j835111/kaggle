#!/usr/bin/env python3
"""里程碑 2：DeBERTa-v3-base 三分類微調（CLI 入口）。

只能在有 torch / transformers 的環境跑（Kaggle Notebook 或有 GPU 的機器）；
本機沒有 GPU，這個檔案沒有、也無法有本機測試覆蓋。實際邏輯在 llmcls/train.py，
這裡只是包一層 argparse，方便在遠端 GPU（例如 RunPod）上用 CLI 跑。
Kaggle Notebook 裡建議直接 `from llmcls.train import train_fold` 呼叫，不要 shell 出來跑這支。

預設只練 fold 0，先確認贏過 scripts/baseline_prior.py 印出的分數，
再決定要不要花時間跑滿 5 folds（掃 --fold 0..4 各跑一次）。

    python scripts/train.py --fold 0 --epochs 2
    python scripts/train.py --fold 0 --predict-test   # 順便對 test.csv 產生 submission
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.config import MAX_LEN, MODEL_NAME, N_FOLDS  # noqa: E402
from llmcls.train import predict_test_and_save, train_fold  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--fold", type=int, default=0, help="只練這個 fold（不是全部 folds）")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="切分用的總 fold 數")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--predict-test", action="store_true", help="順便對 test.csv 推論並寫出 submission")
    args = parser.parse_args()

    result = train_fold(
        fold=args.fold,
        model_name=args.model_name,
        n_folds=args.folds,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )

    if args.predict_test:
        path = predict_test_and_save(result, name=f"submission_fold{args.fold}.csv")
        print(f"提交檔已寫入並通過格式檢查：{path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
