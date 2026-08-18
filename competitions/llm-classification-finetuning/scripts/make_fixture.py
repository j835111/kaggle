#!/usr/bin/env python3
"""產生一份**合成的**小型 fixture 資料，模仿競賽 CSV 的 schema。

用途：在真實資料下載前，讓 CV 切分、log loss、提交組裝這條 pipeline 能在本機跑通。
這不是真實資料，不要拿來訓練或評估模型品質。

    python scripts/make_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.config import LABEL_COLS, SEED  # noqa: E402

FIXTURE_DIR = COMP_ROOT / "data" / "fixture"
N_TRAIN = 600
N_TEST = 50
N_UNIQUE_PROMPTS = 200  # 刻意少於 N_TRAIN，製造重複 prompt 來測試 group 切分
MODELS = ["model_alpha", "model_beta", "model_gamma", "model_delta"]
# 三類的比例大致模仿真實分佈：a/b 略多於 tie。
CLASS_PROBS = [0.35, 0.35, 0.30]


def _turns(rng: np.random.Generator, tag: str, idx: int) -> str:
    """組出一格 JSON 編碼的多輪對話字串。"""
    n_turns = int(rng.integers(1, 3))
    return json.dumps([f"[fixture] {tag} #{idx} turn {t}" for t in range(n_turns)])


def main() -> int:
    rng = np.random.default_rng(SEED)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    prompt_ids = rng.integers(0, N_UNIQUE_PROMPTS, size=N_TRAIN)
    labels = rng.choice(len(LABEL_COLS), size=N_TRAIN, p=CLASS_PROBS)

    train = pd.DataFrame(
        {
            "id": np.arange(1, N_TRAIN + 1),
            "model_a": rng.choice(MODELS, size=N_TRAIN),
            "model_b": rng.choice(MODELS, size=N_TRAIN),
            "prompt": [json.dumps([f"[fixture] prompt #{p}"]) for p in prompt_ids],
            "response_a": [_turns(rng, "response_a", i) for i in range(N_TRAIN)],
            "response_b": [_turns(rng, "response_b", i) for i in range(N_TRAIN)],
        }
    )
    for c, col in enumerate(LABEL_COLS):
        train[col] = (labels == c).astype(int)

    test = pd.DataFrame(
        {
            "id": np.arange(10_001, 10_001 + N_TEST),
            "prompt": [json.dumps([f"[fixture] test prompt #{i}"]) for i in range(N_TEST)],
            "response_a": [_turns(rng, "test response_a", i) for i in range(N_TEST)],
            "response_b": [_turns(rng, "test response_b", i) for i in range(N_TEST)],
        }
    )

    sample = pd.DataFrame({"id": test["id"]})
    for col in LABEL_COLS:
        sample[col] = 1.0 / len(LABEL_COLS)

    train.to_csv(FIXTURE_DIR / "train.csv", index=False)
    test.to_csv(FIXTURE_DIR / "test.csv", index=False)
    sample.to_csv(FIXTURE_DIR / "sample_submission.csv", index=False)

    print(f"已寫入合成 fixture 至 {FIXTURE_DIR}")
    print(f"  train.csv             {len(train)} 列 / {len(set(prompt_ids))} 個唯一 prompt")
    print(f"  test.csv              {len(test)} 列")
    print(f"  sample_submission.csv {len(sample)} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
