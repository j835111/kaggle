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
import os
import shutil
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
    set_seed,
)

from llmcls.config import MAX_LEN, MODEL_DIR, MODEL_NAME, N_CLASSES, N_FOLDS, SEED
from llmcls.cv import add_folds, fold_indices
from llmcls.data import load_test, load_train
from llmcls.metrics import UNIFORM_LOGLOSS, log_loss, softmax
from llmcls.submission import build_submission, save_submission
from llmcls.text import build_input_ids, swap_ab_label
from llmcls.training_safety import should_stop_for_nonfinite
from llmcls.tta import average_swapped


class PreferenceDataset(torch.utils.data.Dataset):
    """把三欄文字 tokenize 成單一序列：[CLS] prompt [SEP] response_a [SEP] response_b [SEP]。"""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int,
        labels: np.ndarray | None,
        ab_swap_prob: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.ab_swap_prob = ab_swap_prob
        # 訓練時 a/b 對調增強（milestone 3）：獨立的 RNG，不動 Python/numpy 的全域
        # 亂數狀態——Trainer 自己的資料洗牌也是靠全域亂數狀態，兩邊共用同一個全域
        # 產生器的話，開不開這個選項會連帶改變洗牌順序，A/B 比較就不乾淨了。固定
        # 用 SEED 起始，同一個 seed 重跑兩次會抽到同一組對調決定。
        self._rng = np.random.default_rng(SEED) if ab_swap_prob > 0 else None
        # 預先 encode 三欄一次，__getitem__ 只做截斷 + 拼接，不用每個 epoch 重複 tokenize。
        # 整批呼叫 tokenizer(...)，不要逐列呼叫 tokenizer.encode()——fast tokenizer 的
        # 平行化是在批次呼叫內部做的（Rust 那邊自己開執行緒），逐列呼叫每次都要重新付一次
        # Python↔Rust FFI 的固定開銷，資料量上萬列時這筆開銷不能忽略，TTA 又是同一批文字
        # 重新 tokenize 兩次（正常順序 + 對調順序），批次呼叫能把這個成本壓下去。
        self.prompt_ids = tokenizer(df["prompt_text"].tolist(), add_special_tokens=False)["input_ids"]
        self.response_a_ids = tokenizer(df["response_a_text"].tolist(), add_special_tokens=False)["input_ids"]
        self.response_b_ids = tokenizer(df["response_b_text"].tolist(), add_special_tokens=False)["input_ids"]

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def __getitem__(self, idx: int) -> dict:
        a_ids, b_ids = self.response_a_ids[idx], self.response_b_ids[idx]
        label = int(self.labels[idx]) if self.labels is not None else None
        # 每次被抓取都重新擲一次骰子（不是固定對調某一半資料）——同一列資料在不同
        # epoch 可能拿到不同順序，訓練久了每一列平均都看過兩種順序。只有訓練集會
        # 傳非 0 的 ab_swap_prob，驗證集固定用原始順序，score 才能跟沒開這個選項
        # 的訓練直接比較。
        if self._rng is not None and self._rng.random() < self.ab_swap_prob:
            a_ids, b_ids = b_ids, a_ids
            if label is not None:
                label = swap_ab_label(label)
        p, a, b = build_input_ids(self.prompt_ids[idx], a_ids, b_ids, self.max_len)
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        input_ids = [cls_id, *p, sep_id, *a, sep_id, *b, sep_id]
        item = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
        if label is not None:
            item["labels"] = label
        return item


class StopOnNonFiniteLoss(TrainerCallback):
    """DeBERTa-v3 在 fp32、peak LR 附近實測會突然發散：loss 衝高、grad_norm 變 NaN，
    之後每一步都是壞的，權重永久壞掉但 Trainer 完全不知道、還是把剩下的 epoch 跑完
    （實測浪費了 83 分鐘 GPU 時間裡的 70 分鐘）。這裡一偵測到就叫它停，把剩下的時間
    省下來，`triggered` 讓呼叫端知道這次訓練發散過、權重不可信。

    只看單一次 log 的 grad_norm 曾經誤判過：fp16 下 GradScaler 遇到某一步梯度
    溢位會自動跳過那次更新、調低 scale factor 再繼續，這是正常現象，loss 本身
    仍然健康，不代表訓練壞掉——實測 group_by_length 那次「發散」在觸發停止的
    那一行 loss 是 1.079（跟前面每一步一樣正常），只有 grad_norm 是 inf，判定
    「有害」其實是這個過度敏感的舊邏輯誤觸發。真正的判斷邏輯（loss 非有限值
    立刻停；grad_norm 非有限值要連續 `grad_norm_patience` 次才算真的卡住）在
    `llmcls.training_safety.should_stop_for_nonfinite()`，本機有測試覆蓋。
    """

    def __init__(self, grad_norm_patience: int = 3) -> None:
        self.triggered = False
        self.grad_norm_patience = grad_norm_patience
        self._consecutive_nonfinite_grad_norm = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            should_stop, self._consecutive_nonfinite_grad_norm = should_stop_for_nonfinite(
                logs, self._consecutive_nonfinite_grad_norm, self.grad_norm_patience
            )
            if should_stop:
                control.should_training_stop = True
                self.triggered = True
        return control


# log_loss 不可能自然達到的高值，訓練中途的 log 看到這個數字就知道那次週期性
# 評估算在非有限值上，是壞掉的資料點，不是真的分數。
_NONFINITE_SENTINEL = 99.0


def _compute_metrics(eval_pred) -> dict:
    """log_loss() 對非法機率值嚴格報錯（這是它的正確行為，見 metrics.py）；但這裡是
    Trainer 的 eval callback，一炸整個 run 就白跑、連權重都存不到，所以不能用 raise。

    早期版本把非有限值夾到均勻機率再算分——結果發散的 checkpoint 剛好算出
    log loss = ln(3)，跟真的健康、剛好打平基準的 checkpoint 混在一起分不出來。
    改成回報一個大到不可能自然出現的哨兵值，訓練中途的 log 一看就知道那個點壞了。
    """
    logits, labels = eval_pred
    probs = softmax(np.asarray(logits))
    finite = np.isfinite(probs).all(axis=1)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        return {"log_loss": _NONFINITE_SENTINEL, "vs_uniform": float("nan"), "n_nonfinite": n_nonfinite}
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
    fp16: bool = True,
    label_smoothing: float = 0.0,
    group_by_length: bool = False,
) -> TrainingArguments:
    # transformers 把 evaluation_strategy 改名成 eval_strategy 過；Kaggle Notebook 內建的
    # 版本不固定，用 inspect 挑對的參數名比硬編一個更穩。
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    # 一律用 steps（不是 epoch）當 eval/save 的節奏，完整訓練也一樣 —— 中途的週期性
    # 存檔是拿來在 kernel 中途出狀況時當復原點用的，1500 是給完整訓練用的預設值：
    # 跟 eval_subset_rows 搭配（train_fold 會把訓練中途的評估換成子集），拉開頻率
    # 不會犧牲發散偵測 —— 那是每 50 步看 loss/grad_norm 的 StopOnNonFiniteLoss
    # callback 在管，跟這裡的 eval 節奏無關。
    #
    # 不用 load_best_model_at_end：實測踩到一個問題——同樣的設定重跑兩次，
    # `load_best_model_at_end` 靠訓練中途對 eval_subset_rows（2000 筆）子集算出來
    # 的分數去挑「最佳」checkpoint，這個子集本身雜訊就不小，兩次重跑各自挑到不同
    # 進度的 checkpoint 當最終權重（實測分別挑中 epoch 1.049 跟 epoch 1.574），
    # 光是這個選擇上的雜訊就足以讓兩次「應該一樣」的最終分數飄動超過 0.01——跟
    # label smoothing/a、b 對調增強量到的效果量同一個量級，會讓 A/B 比較失去意義。
    # 子集分數只拿來在訓練中途看趨勢（原本的設計目的），不該拿來決定「用哪個版本
    # 的權重」；固定用訓練跑完當下的最終狀態，才不會多引入這層雜訊。
    default_eval_steps = max(1, max_steps // 2) if max_steps is not None else 1500
    steps = eval_steps or default_eval_steps
    kwargs = dict(
        output_dir=str(output_dir),
        **{strategy_key: "steps"},
        save_strategy="steps",
        eval_steps=steps,
        save_steps=steps,
        save_total_limit=1,
        load_best_model_at_end=False,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        # 第一次踩到的坑是模型「裸」用 fp16（checkpoint 原本就存 fp16，from_pretrained
        # 沒有轉型），完全沒有 loss scaler 保護。train_fold() 現在會先強制 model.float()
        # 轉成真正的 fp32，這裡的 fp16=True 才是正規流程：autocast 動態轉型 + GradScaler
        # 做 loss scaling，梯度真的非有限值時 GradScaler 會跳過那一步而不是把權重弄壞，
        # StopOnNonFiniteLoss 則是最後一道防線。T4 有 fp16 tensor core，這樣才吃得到
        # 混合精度的加速。
        fp16=fp16,
        # Trainer 內建的 label smoothing：labels 還是整數類別（不用先轉成 one-hot），
        # HF 的 LabelSmoother 會在算 cross entropy 時自動把目標機率從 1.0 壓低、
        # 分一點出去給另外兩類。0.0 等於關閉，行為跟原本完全一樣。
        label_smoothing_factor=label_smoothing,
        # 把長度相近的樣本分到同一個 batch，減少 padding 浪費的算力。PreferenceDataset
        # 不是 datasets.Dataset，Trainer 會退回自己對每一筆呼叫 len(item["input_ids"])
        # 來排序（PreferenceDataset.__getitem__ 回傳的就是這個 key，天生相容，不用額外
        # 接一個 length 欄位）。風險見 train_fold() 的 group_by_length 說明。
        group_by_length=group_by_length,
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


def predict_logits(trainer: Trainer, tokenizer, df: pd.DataFrame, max_len: int) -> np.ndarray:
    """對沒有 label 的 DataFrame（例如 test set）跑推論，回傳 (n, N_CLASSES) 的原始 logits
    （softmax 之前）——temperature scaling 要在這個尺度上配溫度，不能對已經 softmax
    過的機率配。
    """
    ds = PreferenceDataset(df, tokenizer, max_len, labels=None)
    return np.asarray(trainer.predict(ds).predictions)


def predict_probs(trainer: Trainer, tokenizer, df: pd.DataFrame, max_len: int) -> np.ndarray:
    """對沒有 label 的 DataFrame（例如 test set）跑推論，回傳 (n, N_CLASSES) 機率。"""
    return softmax(predict_logits(trainer, tokenizer, df, max_len))


def swap_ab(df: pd.DataFrame) -> pd.DataFrame:
    """回傳 response_a_text / response_b_text 對調後的複本，其餘欄位不變 ——
    PreferenceDataset 只讀這兩欄跟 prompt_text 建輸入，對調這兩欄就等於把
    response_a / response_b 的順序整個倒過來重新推論一次。
    """
    swapped = df.copy()
    swapped["response_a_text"] = df["response_b_text"].to_numpy()
    swapped["response_b_text"] = df["response_a_text"].to_numpy()
    return swapped


def train_fold(
    fold: int = 0,
    model_name: str = MODEL_NAME,
    n_folds: int = N_FOLDS,
    max_len: int = MAX_LEN,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-5,
    lr_scheduler_type: str = "linear",
    fp16: bool = True,
    output_dir: Path | None = None,
    max_train_rows: int | None = None,
    max_valid_rows: int | None = None,
    max_steps: int | None = None,
    eval_steps: int | None = None,
    eval_subset_rows: int | None = None,
    label_smoothing: float = 0.0,
    group_by_length: bool = False,
    ab_swap_prob: float = 0.0,
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

    `eval_subset_rows` 是完整訓練用的加速選項：訓練中途的週期性評估只在這個子集上跑
    （原本每次評估都對完整驗證集跑一次，實測光是評估就佔掉總訓練時間近一半），最後
    收斂完仍然會對完整驗證集重新 `evaluate()` 一次，回傳的 `score` 保證是完整驗證集
    的分數，不會被子集的雜訊污染。

    `label_smoothing`（milestone 3）：0.0 是關閉，跟原本行為一樣；HF Trainer 內建
    支援，不用自己改 labels 或 loss function。實測 0.1 讓 fold 0 valid log loss
    從 1.06840 變差成 1.08815，已經放棄，不要再試（見 README.md）。

    `group_by_length`：把長度相近的樣本分到同一個 batch，減少 padding 浪費。
    False 是關閉，跟原本行為一樣。風險：這正好會把最長的句子集中到同一批，獨立的
    profiling kernel 已經測過最壞情況（fold 0 最長 16 筆組成一個 batch）單步
    forward+backward 不會 OOM（餘裕 23.6%），但那是單步測試，完整一個 epoch 訓練
    下來記憶體碎片化累積會不會更緊繃還沒驗證過，第一次用這個設定時不要跳過
    `diverged`/`n_nonfinite` 的檢查。

    `ab_swap_prob`（milestone 3，訓練時 a/b 對調增強）：訓練集每一筆資料在每次
    被 `PreferenceDataset.__getitem__` 抓取時，有這個機率被動態對調
    response_a/response_b（連同標籤一起用 `swap_ab_label()` 對調：0⟷1，2 不變），
    每個 epoch 重新擲一次骰子，同一列在不同 epoch 可能拿到不同順序。只套用在
    訓練集，驗證集固定用原始順序不受影響，`score` 才能跟沒開這個選項的訓練直接
    比較。0.0 是關閉，跟原本行為一樣。動機：winner_tie 診斷（見
    `calibrate_folds.py`）發現模型對 `winner_model_a`/`winner_model_b` 兩類的
    原始（未做 TTA）log loss 落差很大（1.04710 vs 1.11257），推論時的 TTA 已經
    在事後修正一部分，這裡要測的是訓練時直接解決順序偏見，減少對推論時 TTA
    的依賴。

    `batch_size` 調大要非常小心：煙霧測試只能驗證穩定性（會不會發散），驗不出「完整
    資料集上的記憶體上限」——`max_train_rows` 抽樣的子集很難剛好抽到全是接近
    `max_len` 上限的最壞情況那幾批。實測 batch_size=16 在 3000 筆的煙霧測試上完全
    穩定，換成完整的 45746 筆卻在訓練中途 CUDA OOM（T4 記憶體只差 66MB）。batch_size
    調大之前，煙霧測試過關不代表完整資料集上安全。
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
    # 分類頭（deberta-v3-base 本身沒有，from_pretrained 會隨機初始化一層新的）是在
    # Trainer 建立、TrainingArguments(seed=...) 生效之前就跑掉的——實測同一個 fold
    # 用完全相同的設定重跑，valid log loss 可以飄動 0.01~0.02（跟 label smoothing/
    # group_by_length 那兩次判定「有害」的差距同一個量級），才發現這裡才是真正決定
    # 起始點隨機性的地方，TrainingArguments 的 seed 只固定得了訓練「過程」（資料
    # 洗牌順序、dropout），固定不了「起點」。這裡先呼叫 set_seed() 才能讓同一個
    # seed 重跑兩次得到同一個分類頭初始值，A/B 比較才有意義。
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=N_CLASSES)
    # 實測 deberta-v3-base 在 HF Hub 上是用 fp16 存的，新版 transformers 的
    # from_pretrained 預設照抄 checkpoint 原本的 dtype，跟 TrainingArguments(fp16=False)
    # 完全無關 —— 結果是在跑「沒有 loss scaler 保護的裸 fp16 訓練」，梯度撐不了多久
    # 就溢位成 NaN，調低學習率只是延後發生、不是解法。強制轉 fp32 才是真正對應
    # fp16=False 的意圖。
    print(f"model 載入時的 dtype：{next(model.parameters()).dtype}")
    model = model.float()

    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    # 用隨機抽樣而不是頭幾列 —— 頭幾列在煙霧測試時永遠是同一批，測不到資料的多樣性。
    if max_train_rows is not None:
        tr_df = tr_df.sample(n=min(max_train_rows, len(tr_df)), random_state=SEED).reset_index(drop=True)
    if max_valid_rows is not None:
        va_df = va_df.sample(n=min(max_valid_rows, len(va_df)), random_state=SEED).reset_index(drop=True)
    tr_ds = PreferenceDataset(tr_df, tokenizer, max_len, tr_df["label"].to_numpy(), ab_swap_prob=ab_swap_prob)
    va_ds = PreferenceDataset(va_df, tokenizer, max_len, va_df["label"].to_numpy())

    # 訓練中途的週期性評估用子集（快很多），最後才對完整驗證集重新算一次真正的分數。
    if eval_subset_rows is not None and eval_subset_rows < len(va_df):
        va_df_periodic = va_df.sample(n=eval_subset_rows, random_state=SEED).reset_index(drop=True)
        va_ds_periodic = PreferenceDataset(va_df_periodic, tokenizer, max_len, va_df_periodic["label"].to_numpy())
    else:
        va_ds_periodic = va_ds

    stop_callback = StopOnNonFiniteLoss()
    trainer = Trainer(
        model=model,
        args=_training_args(
            output_dir, epochs, batch_size, lr, lr_scheduler_type=lr_scheduler_type,
            max_steps=max_steps, eval_steps=eval_steps, fp16=fp16,
            label_smoothing=label_smoothing, group_by_length=group_by_length,
        ),
        train_dataset=tr_ds,
        eval_dataset=va_ds_periodic,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[stop_callback],
    )
    trainer.train()
    if stop_callback.triggered:
        print("偵測到 loss/grad_norm 變成 NaN，已提前停止訓練 —— 這次的權重不可信，不要拿去推論")

    # 存檔緊接在 train() 後面、explicit evaluate() 之前 —— 不用 load_best_model_at_end
    # 挑歷史最佳（見 _training_args 的說明），trainer.model 現在就是訓練跑完當下的
    # 最終狀態，這裡先存起來，後面的 evaluate() 就算出狀況也不會白跑一整個 epoch 的
    # GPU 時間卻什麼都沒留下。
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # TrainingArguments 的 output_dir 跟這裡的最終存檔目錄是同一個路徑，Trainer 自己
    # 的 save_steps 週期性存檔（含 optimizer/scheduler state，體積是模型本身的 2-3
    # 倍）還留在 output_dir/checkpoint-*/ 底下，跟上面剛存好的最終權重是同一份東西的
    # 重複備份。單一 fold 時這筆多餘的空間還在 Kaggle 磁碟額度內，5 個 fold 一次跑完
    # 疊起來就會把磁碟塞爆（實測踩到：5 folds 沒清、跑到一半磁碟就滿了）。權重已經
    # 存到 output_dir 頂層，這些子目錄可以直接刪掉。
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint_dir)
    print(f"模型已存到 {output_dir}（訓練中途的 checkpoint-* 已清除）")

    # 明確傳完整的 va_ds —— 訓練中途用的可能是子集，最終回報的分數必須是完整驗證集
    # 算出來的，不能被子集的雜訊污染。
    metrics = trainer.evaluate(eval_dataset=va_ds)
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
    model = AutoModelForSequenceClassification.from_pretrained(str(output_dir)).float()
    return model, tokenizer


def predict_logits_with_model(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int = 32) -> np.ndarray:
    """離線推論 notebook 用：不需要 Trainer / TrainingArguments，直接跑 forward，
    回傳 softmax 之前的原始 logits（temperature scaling 要配在這個尺度上）。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    ds = PreferenceDataset(df, tokenizer, max_len, labels=None)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            inputs = collator(batch).to(device)
            logits = model(**inputs).logits.detach().cpu().numpy()
            all_logits.append(logits)
    return np.concatenate(all_logits, axis=0)


def predict_with_model(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int = 32) -> np.ndarray:
    """離線推論 notebook 用：不需要 Trainer / TrainingArguments，直接跑 forward。"""
    return softmax(predict_logits_with_model(model, tokenizer, df, max_len, batch_size))


def predict_logits_with_tta(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int = 32) -> np.ndarray:
    """a/b 對調 TTA，回傳原始順序與對調順序（已換回欄位對齊）logits 的平均。

    對付位置偏誤：模型可能學到偏好放在 A 或 B 位置本身，而不是回覆的品質。回傳
    logits（不是機率）是為了跟 temperature scaling 串接——要接著配溫度的話，必須
    先在 logits 尺度合併成一組，再對『合併後的 logits』配溫度，在機率層級平均、
    再對已經攤平過的機率配溫度會失真。單純只要 TTA、不接 calibration 的話，直接
    對這裡回傳的結果做 softmax 即可。
    """
    logits_orig = predict_logits_with_model(model, tokenizer, df, max_len, batch_size)
    logits_swapped = predict_logits_with_model(model, tokenizer, swap_ab(df), max_len, batch_size)
    return average_swapped(logits_orig, logits_swapped)


def predict_with_tta(model, tokenizer, df: pd.DataFrame, max_len: int, batch_size: int = 32) -> np.ndarray:
    """a/b 對調 TTA：原始順序跟對調順序各推論一次，換回欄位對齊後在機率層級平均。"""
    probs_orig = predict_with_model(model, tokenizer, df, max_len, batch_size)
    probs_swapped = predict_with_model(model, tokenizer, swap_ab(df), max_len, batch_size)
    return average_swapped(probs_orig, probs_swapped)


def predict_test_and_save(trainer_result: dict, name: str = "submission.csv") -> Path:
    """train_fold() 回傳值直接餵進來，對 test.csv 推論並寫出提交檔。"""
    test = load_test()
    probs = predict_probs(trainer_result["trainer"], trainer_result["tokenizer"], test, MAX_LEN)
    sub = build_submission(test["id"], probs)
    return save_submission(sub, name=name)
