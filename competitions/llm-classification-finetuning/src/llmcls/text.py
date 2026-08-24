"""把 prompt / response_a / response_b 的 token id 組成單一模型輸入序列，
超過 max_len 時做 head+tail 截斷。

刻意在 id 層級操作（呼叫端先分別 encode 三段，這裡只管截斷與長度預算），而不是
組字串再整段重新 tokenize —— 這樣三段可以各自截斷，不會因為 response_a 太長就把
response_b 擠到只剩尾巴一小段，兩邊被模型看到的資訊量比較公平。

不依賴 torch / transformers，純 Python 可在本機測試。真正呼叫 tokenizer 的地方在
llmcls/train.py（那裡才需要 GPU 環境）。
"""

from __future__ import annotations

# 對應 [CLS] prompt [SEP] response_a [SEP] response_b [SEP] 的組法：1 個 CLS + 3 個 SEP。
NUM_SPECIAL_TOKENS = 4

# 三段的預算比例：prompt 通常比兩份回覆短，兩份回覆同等重要所以各拿更多。
DEFAULT_RATIOS = (0.2, 0.4, 0.4)


def truncate_ids(ids: list[int], budget: int, head_ratio: float = 0.5) -> list[int]:
    """留頭尾、砍中間。budget <= 0 回傳空列表，不需要截斷時原樣回傳。"""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    head_len = min(budget, max(1, round(budget * head_ratio)))
    tail_len = budget - head_len
    if tail_len <= 0:
        return ids[:head_len]
    return ids[:head_len] + ids[len(ids) - tail_len :]


def split_budget(total: int, ratios: tuple[float, float, float] = DEFAULT_RATIOS) -> tuple[int, int, int]:
    """依 ratios 把 total 分給三段；四捨五入的誤差全部歸給最後一段，確保三段總和精確等於 total。"""
    if total <= 0:
        return (0, 0, 0)
    a = int(total * ratios[0])
    b = int(total * ratios[1])
    c = total - a - b
    return (a, b, c)


def swap_ab_label(label: int) -> int:
    """訓練時 a/b 對調增強用：response_a/response_b 對調之後，原本「a 贏」
    （0）要變成「b 贏」（1），「b 贏」要變成「a 贏」，「打平」（2）不受影響
    ——這兩類是靠標籤本身的整數編碼互換位置，跟 llmcls.train.swap_ab() 只換
    DataFrame 欄位、不動標籤是兩回事（那個是給推論 TTA 用的，模型輸出的機率
    欄位事後靠 [:, [1, 0, 2]] 換回來對齊，不需要動標籤）。
    """
    return 1 - label if label in (0, 1) else label


def build_input_ids(
    prompt_ids: list[int],
    response_a_ids: list[int],
    response_b_ids: list[int],
    max_len: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    head_ratio: float = 0.5,
) -> tuple[list[int], list[int], list[int]]:
    """回傳截斷後的 (prompt_ids, response_a_ids, response_b_ids)。

    呼叫端還要自己補上 CLS/SEP 特殊 token，所以保證
    len(p) + len(a) + len(b) + NUM_SPECIAL_TOKENS <= max_len。
    """
    budget = max_len - NUM_SPECIAL_TOKENS
    if budget <= 0:
        raise ValueError(f"max_len ({max_len}) 太小，容不下 {NUM_SPECIAL_TOKENS} 個特殊 token")
    p_budget, a_budget, b_budget = split_budget(budget, ratios)
    return (
        truncate_ids(prompt_ids, p_budget, head_ratio),
        truncate_ids(response_a_ids, a_budget, head_ratio),
        truncate_ids(response_b_ids, b_budget, head_ratio),
    )
