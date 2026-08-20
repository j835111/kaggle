"""DeBERTa-v3-base 三分類微調：資料集組裝、訓練、推論。

只能在有 torch / transformers 的環境 import（Kaggle Notebook 或有 GPU 的機器）；
本機沒有 GPU，這個檔案沒有、也無法有本機測試覆蓋。截斷邏輯本身在 llmcls/text.py
裡用純 Python 測試過，這裡只是把它接上真正的 tokenizer 和 HF Trainer。

`train_fold()` 是核心入口，同時給兩種呼叫方式用：
- CLI：scripts/train.py（適合遠端 GPU，例如 RunPod）
- Kaggle Notebook cell：`from llmcls.train import train_fold` 直接呼叫
"""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from llmcls.config import MAX_LEN, MODEL_DIR, MODEL_NAME, N_CLASSES, N_FOLDS, SEED
from llmcls.cv import add_folds, fold_indices
from llmcls.data import load_test, load_train
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss
from llmcls.submission import build_submission, save_submission
from llmcls.text import build_input_ids


class PreferenceDataset(torch.utils.data.Dataset):
    """把三欄文字 tokenize 成單一序列：[CLS] prompt [SEP] response_a [SEP] response_b [SEP]。"""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, labels: np.ndarray | None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        # 預先 encode 三欄一次，__getitem__ 只做截斷 + 拼接，不用每個 epoch 重複 tokenize。
        self.prompt_ids = [tokenizer.encode(t, add_special_tokens=False) for t in df["prompt_text"]]
        self.response_a_ids = [tokenizer.encode(t, add_special_tokens=False) for t in df["response_a_text"]]
        self.response_b_ids = [tokenizer.encode(t, add_special_tokens=False) for t in df["response_b_text"]]

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def __getitem__(self, idx: int) -> dict:
        p, a, b = build_input_ids(
            self.prompt_ids[idx], self.response_a_ids[idx], self.response_b_ids[idx], self.max_len
        )
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        input_ids = [cls_id, *p, sep_id, *a, sep_id, *b, sep_id]
        item = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


class StopOnNonFiniteLoss(TrainerCallback):
    """DeBERTa-v3 在 fp32、peak LR 附近實測會突然發散：loss 衝高、grad_norm 變 NaN，
    之後每一步都是壞的，權重永久壞掉但 Trainer 完全不知道、還是把剩下的 epoch 跑完
    （實測浪費了 83 分鐘 GPU 時間裡的 70 分鐘）。這裡一偵測到就叫它停，把剩下的時間
    省下來，`triggered` 讓呼叫端知道這次訓練發散過、權重不可信。
    """

    def __init__(self) -> None:
        self.triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and any(
            isinstance(v, (int, float)) and not math.isfinite(v) for k, v in logs.items() if k in ("loss", "grad_norm")
        ):
            control.should_training_stop = True
            self.triggered = True
        return control


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _compute_metrics(eval_pred) -> dict:
    """log_loss() 對非法機率值嚴格報錯（這是它的正確行為，見 metrics.py）；但這裡是
    Trainer 的 eval callback，一炸整個 run 就白跑、連權重都存不到。改成統計有幾列非
    有限值、夾到均勻機率再算分，異常本身用 n_nonfinite 回報，不讓它摧毀整次訓練。
    """
    logits, labels = eval_pred
    probs = softmax(np.asarray(logits))
    finite = np.isfinite(probs).all(axis=1)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        probs = probs.copy()
        probs[~finite] = 1.0 / N_CLASSES
    score = log_loss(labels, probs)
    return {"log_loss": score, "vs_uniform": UNIFORM_LOGLOSS - score, "n_nonfinite": n_nonfinite}


def _training_args(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_scheduler_type: str = "linear",
    max_steps: int | None = None,
    eval_steps: int | None = None,
) -> TrainingArguments:
    # transformers 把 evaluation_strategy 改名成 eval_strategy 過；Kaggle Notebook 內建的
    # 版本不固定，用 inspect 挑對的參數名比硬編一個更穩。
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    # 一律用 steps（不是 epoch）當 eval/save 的節奏，完整訓練也一樣 —— 實測 DeBERTa-v3
    # 在 fp32 訓練到一半會發散，只在 epoch 邊界存檔的話，發散前那個還健康的檢查點根本
    # 沒機會被存下來，load_best_model_at_end 也就沒有東西可挑。
    default_eval_steps = max(1, max_steps // 2) if max_steps is not None else 500
    steps = eval_steps or default_eval_steps
    kwargs = dict(
        output_dir=str(output_dir),
        **{strategy_key: "steps"},
        save_strategy="steps",
        eval_steps=steps,
        save_steps=steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        # fp16 混合精度在 Kaggle 目前的 transformers/accelerate 組合下，DeBERTa-v3
        # 第一次 backward 就炸「Attempting to unscale FP16 gradients」（已知相容性
        # 問題）。T4 記憶體對 deberta-v3-base + batch_size 8 + max_len 512 綽綽有餘，
        # 先用 fp32 求正確跑通，混合精度是效能優化、不是里程碑 2 的目標。
        fp16=False,
        report_to=[],
        logging_steps=10 if max_steps else 50,
        # 關掉 tqdm 進度條、強制用純文字 print 記錄 loss —— Kaggle Notebook 預設會用
        # rich/widget 進度條，那些輸出只進 __notebook__.ipynb 的 cell output，不會出現
        # 在 `kaggle kernels output` 抓得到的純文字 log，事後完全查不到 loss 曲線。
        disable_tqdm=True,
        seed=42,
    )
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return TrainingArguments(**kwargs)


def predict_probs(trainer: Trainer, tokenizer, df: pd.DataFrame, max_len: int) -> np.ndarray:
    """對沒有 label 的 DataFrame（例如 test set）跑推論，回傳 (n, N_CLASSES) 機率。"""
    ds = PreferenceDataset(df, tokenizer, max_len, labels=None)
    logits = trainer.predict(ds).predictions
    return softmax(np.asarray(logits))


def train_fold(
    fold: int = 0,
    model_name: str = MODEL_NAME,
    n_folds: int = N_FOLDS,
    max_len: int = MAX_LEN,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    lr_scheduler_type: str = "linear",
    output_dir: Path | None = None,
    max_train_rows: int | None = None,
    max_valid_rows: int | None = None,
    max_steps: int | None = None,
    eval_steps: int | None = None,
) -> dict:
    """練一個 fold，存權重，回傳 {"score", "n_nonfinite", "diverged", "output_dir",
    "trainer", "tokenizer"}。

    預設只練 fold 0，不是全部 n_folds —— 先確認贏過 baseline_prior.py 印出的分數，
    再決定要不要花時間跑滿整個 CV。

    `max_train_rows` / `max_valid_rows` / `max_steps` / `eval_steps` 是煙霧測試用的：
    隨機抽一小撮資料、跑幾十步就評估一次，把「資料→tokenize→forward→eval→存檔」整條
    路徑在幾分鐘內走過一遍，而不是每次改動都要賭一整個 epoch（30-60 分鐘 GPU 時間）
    才知道炸不炸。跑煙霧測試時務必把 `lr_scheduler_type` 設成 "constant_with_warmup"
    ——用預設的 "linear" 配上 `max_steps` 很小的話，學習率暖身完就立刻開始衰減，
    根本沒有停留在 peak LR 的時間，測不出「訓練到 peak LR 附近才發散」這種問題
    （這正是本專案第一次煙霧測試沒抓到、完整訓練卻在 peak LR 附近整個發散的原因）。
    """
    # Kaggle 的 GPU kernel 預設給 T4 x2；HF Trainer 偵測到多張卡會自動包成
    # nn.DataParallel，這是已知會在 eval 階段的 predictions gather 上出怪問題的來源
    # （一個訊號：「gather along dimension 0 ... all input tensors were scalars」的
    # warning）。deberta-v3-base 在 batch_size 8 / max_len 512 下單張 T4 就跑得動，
    # 沒有 DP 帶來的好處，直接限制成單卡排除這個變因。用 setdefault 而不是強制覆蓋，
    # 呼叫端仍可自行指定 CUDA_VISIBLE_DEVICES。
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    output_dir = Path(output_dir) if output_dir is not None else MODEL_DIR / f"fold{fold}"

    train = load_train()
    print(f"train: {len(train)} 列")
    train = add_folds(train, n_folds=n_folds)
    tr_idx, va_idx = fold_indices(train, fold)
    print(f"fold {fold}: train {len(tr_idx)} 列, valid {len(va_idx)} 列")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=N_CLASSES)

    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    # 用隨機抽樣而不是頭幾列 —— 頭幾列在煙霧測試時永遠是同一批，測不到資料的多樣性。
    if max_train_rows is not None:
        tr_df = tr_df.sample(n=min(max_train_rows, len(tr_df)), random_state=SEED).reset_index(drop=True)
    if max_valid_rows is not None:
        va_df = va_df.sample(n=min(max_valid_rows, len(va_df)), random_state=SEED).reset_index(drop=True)
    tr_ds = PreferenceDataset(tr_df, tokenizer, max_len, tr_df["label"].to_numpy())
    va_ds = PreferenceDataset(va_df, tokenizer, max_len, va_df["label"].to_numpy())

    stop_callback = StopOnNonFiniteLoss()
    trainer = Trainer(
        model=model,
        args=_training_args(
            output_dir, epochs, batch_size, lr, lr_scheduler_type=lr_scheduler_type,
            max_steps=max_steps, eval_steps=eval_steps,
        ),
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[stop_callback],
    )
    trainer.train()
    if stop_callback.triggered:
        print("偵測到 loss/grad_norm 變成 NaN，已提前停止訓練 —— 這次的權重不可信，不要拿去推論")

    # 存檔緊接在 train() 後面、explicit evaluate() 之前 —— load_best_model_at_end=True
    # 已經把最佳權重換回 trainer.model，這裡先存起來，後面的 evaluate() 就算出狀況
    # 也不會白跑一整個 epoch 的 GPU 時間卻什麼都沒留下。
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"模型已存到 {output_dir}")

    metrics = trainer.evaluate()
    score = metrics["eval_log_loss"]
    n_nonfinite = metrics.get("eval_n_nonfinite", 0)
    delta = UNIFORM_LOGLOSS - score
    # n_nonfinite > 0 代表分數是拿均勻機率湊出來的假象（例如全部 clamp 之後 delta 剛好
    # 等於 0，會被誤判成「打平基準」）——只要有非有限值，不管 delta 多少一律算沒過關。
    passed = n_nonfinite == 0 and not stop_callback.triggered and delta > 0
    print(f"\nvalid log loss   {score:.5f}")
    print(f"均勻亂猜基準      {UNIFORM_LOGLOSS:.5f}  (ln 3)")
    print(f"改善              {delta:+.5f}  {'✓ 優於基準' if passed else '✗ 未優於基準'}")
    if n_nonfinite:
        print(f"警告：{n_nonfinite}/{len(va_df)} 筆驗證預測是 NaN/inf，已夾到均勻機率計分 —— 分數不可信，先查訓練穩定性")

    return {
        "score": score,
        "n_nonfinite": n_nonfinite,
        "diverged": stop_callback.triggered,
        "output_dir": output_dir,
        "trainer": trainer,
        "tokenizer": tokenizer,
    }


def load_trained(output_dir: Path):
    """從已存的權重目錄載入 model + tokenizer（供推論 notebook 用，不需要 Trainer）。"""
    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))
    return model, tokenizer


def predict_with_model(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int = 32) -> np.ndarray:
    """離線推論 notebook 用：不需要 Trainer / TrainingArguments，直接跑 forward。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    ds = PreferenceDataset(df, tokenizer, max_len, labels=None)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            inputs = collator(batch).to(device)
            logits = model(**inputs).logits.detach().cpu().numpy()
            all_probs.append(softmax(logits))
    return np.concatenate(all_probs, axis=0)


def predict_test_and_save(trainer_result: dict, name: str = "submission.csv") -> Path:
    """train_fold() 回傳值直接餵進來，對 test.csv 推論並寫出提交檔。"""
    test = load_test()
    probs = predict_probs(trainer_result["trainer"], trainer_result["tokenizer"], test, MAX_LEN)
    sub = build_submission(test["id"], probs)
    return save_submission(sub, name=name)
