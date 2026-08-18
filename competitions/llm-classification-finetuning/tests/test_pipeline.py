"""針對本機這條 pipeline 的單元測試（不碰 torch / transformers）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COMP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMP_ROOT / "src"))

from llmcls.cv import add_folds, prompt_group_key
from llmcls.data import load_train, parse_turns
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss
from llmcls.submission import build_submission, validate_submission

FIXTURE_DIR = COMP_ROOT / "data" / "fixture"


@pytest.fixture(scope="module")
def fixture_dir() -> Path:
    if not (FIXTURE_DIR / "train.csv").exists():
        subprocess.run([sys.executable, str(COMP_ROOT / "scripts" / "make_fixture.py")], check=True)
    return FIXTURE_DIR


# ---- parse_turns：真實資料是 JSON list，但要能容忍壞格式 ----


def test_parse_turns_json_list():
    assert parse_turns(json.dumps(["a", "b"])) == ["a", "b"]


def test_parse_turns_null_element():
    assert parse_turns('["a", null]') == ["a", ""]


def test_parse_turns_plain_text_fallback():
    assert parse_turns("not json at all") == ["not json at all"]


def test_parse_turns_missing():
    assert parse_turns(None) == []
    assert parse_turns(float("nan")) == []
    # pandas 3.x 的缺值可能是 pd.NA 而非 float nan —— 不能讓它掉進 fallback 路徑
    assert parse_turns(pd.NA) == []


def test_load_train_raises_when_column_is_not_json(tmp_path):
    """schema 假設錯誤時要當場報錯，而不是靜默把原始字串當成文字餵進模型。"""
    bad = tmp_path / "train.csv"
    n = 50
    pd.DataFrame(
        {
            "id": range(n),
            "prompt": ["這根本不是 JSON"] * n,
            "response_a": ['["a"]'] * n,
            "response_b": ['["b"]'] * n,
            "winner_model_a": [1] * n,
            "winner_model_b": [0] * n,
            "winner_tie": [0] * n,
        }
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="schema 可能與預期不符"):
        load_train(bad)


# ---- log loss ----


def test_uniform_prediction_equals_ln3():
    y = np.array([0, 1, 2])
    p = np.full((3, 3), 1 / 3)
    assert log_loss(y, p) == pytest.approx(UNIFORM_LOGLOSS)


def test_confident_correct_beats_uniform():
    y = np.array([0, 1, 2])
    p = np.eye(3) * 0.8 + 0.1
    assert log_loss(y, p) < UNIFORM_LOGLOSS


def test_log_loss_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        log_loss(np.array([0]), np.array([[np.nan, 0.5, 0.5]]))


# ---- CV：核心不變量是「同一 prompt 不跨 fold」 ----


def test_folds_have_no_prompt_leak(fixture_dir):
    df = add_folds(load_train(fixture_dir / "train.csv"), n_folds=5)
    groups = prompt_group_key(df)
    per_group = pd.DataFrame({"g": groups.to_numpy(), "f": df["fold"].to_numpy()})
    assert per_group.groupby("g")["f"].nunique().max() == 1
    assert sorted(df["fold"].unique()) == [0, 1, 2, 3, 4]


# ---- 提交檔驗證 ----


def test_valid_submission_passes():
    sub = build_submission([1, 2], np.full((2, 3), 1 / 3))
    validate_submission(sub, sample_path=Path("/nonexistent"))


def test_submission_rejects_rows_not_summing_to_one():
    sub = build_submission([1], np.array([[0.5, 0.2, 0.2]]))
    with pytest.raises(ValueError, match="機率和"):
        validate_submission(sub, sample_path=Path("/nonexistent"))


def test_submission_rejects_duplicate_ids():
    sub = build_submission([1, 1], np.full((2, 3), 1 / 3))
    with pytest.raises(ValueError, match="重複"):
        validate_submission(sub, sample_path=Path("/nonexistent"))


def test_submission_id_set_must_match_sample(fixture_dir):
    sub = build_submission([999_999], np.full((1, 3), 1 / 3))
    with pytest.raises(ValueError, match="sample_submission"):
        validate_submission(sub, sample_path=fixture_dir / "sample_submission.csv")


# ---- 資料載入的 schema 斷言 ----


def test_load_train_rejects_missing_columns(tmp_path):
    bad = tmp_path / "train.csv"
    pd.DataFrame({"id": [1], "prompt": ["x"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="缺少預期欄位"):
        load_train(bad)


def test_load_train_rejects_bad_onehot(tmp_path):
    bad = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "id": [1],
            "prompt": ['["p"]'],
            "response_a": ['["a"]'],
            "response_b": ['["b"]'],
            "winner_model_a": [1],
            "winner_model_b": [1],
            "winner_tie": [0],
        }
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="恰好一個 1"):
        load_train(bad)
