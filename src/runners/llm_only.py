# src/runners/llm_only.py
# -*- coding: utf-8 -*-
"""
LLM-only 测试集生成入口：
- 输入：一句话/短段需求描述 (--desc)
- 输出：统一表格 (parquet + csv)，包含 BASE / SYN / NOISE / SLANG / DIALECT / TYPO / CTX / SAFETY 等
- 逻辑：完全依赖 LLM 生成（不走模板/插槽），再做轻量清洗与强去重

用法示例：
python -m src.runners.llm_only \
  --config configs/agent.yaml \
  --desc "智能家居语音助手，控制灯光/空调/扫地机器人，并支持闲聊与安全测试" \
  --out data/generated/home_llm/cases.parquet \
  --total 300
"""
import argparse
import os
import re
import json
import uuid
import pandas as pd
import yaml

from ..chains import llm_generators as LG


# =========================
# 清洗 & 去重（项目内独立实现）
# =========================

_MID_DOT_CHARS = "·•∙⋅・｡．●"
_FULLWIDTH_PUNCS = "，。！？；：（）【】「」『』“”‘’／＼"
_ASCII_PUNCS = ",.!?;:()[]\"'/-\\"

# 标点统一映射（可按需补充）
_PUNC_MAP = {
    "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
    "（": "(", "）": ")", "【": "[", "】": "]",
    "「": "\"", "」": "\"", "『": "\"", "』": "\"",
    "“": "\"", "”": "\"", "‘": "'", "’": "'",
    "／": "/", "＼": "\\", "－": "-", "—": "-",
}

# 允许的“轻口语”前后缀（不参与去重键生成时会被清除）
_SOFT_FILLERS = [
    "嗯", "呃", "额", "那个", "然后", "请问", "拜托", "麻烦你", "劳驾", "可以不", "能不能",
    "那个啥", "多谢", "谢谢", "辛苦了", "好嘛", "好吗", "好吧", "好不", "呗", "啦",
]

def _strip_soft_fillers(s: str) -> str:
    s = s.strip()
    # 去掉开头/结尾一些客套或口头禅（用于去重键，不会改动原始 query）
    # 注意：这里只做极轻量匹配，避免过度清洗
    for w in _SOFT_FILLERS:
        if s.startswith(w):
            s = s[len(w):].strip()
        if s.endswith(w):
            s = s[:-len(w)].strip()
    return s

def normalize_query(s: str) -> str:
    """归一化：去中点、全半角统一、空白归一、数字/字母的全角->半角（可按需扩展）"""
    if not isinstance(s, str):
        return s

    # 删除中点类字符
    s = re.sub(f"[{re.escape(_MID_DOT_CHARS)}]", "", s)

    # 全角标点 -> 半角
    for k, v in _PUNC_MAP.items():
        s = s.replace(k, v)

    # 全角空格 -> 半角；多空格 -> 单空格
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)

    # 去掉首尾空白
    s = s.strip()
    return s

def dedup_records(records):
    """
    以“句子本身”为去重键：
    - 完全相同去重
    - 仅标点/中点不同视为同句
    - 轻量去除开头/结尾客套词用于生成键，但不会改写原句
    """
    seen = set()
    kept = []
    for r in records:
        q = r.get("query") or ""
        q_norm = normalize_query(q)
        # 构造去重键：再额外去掉轻口语前后缀
        key = _strip_soft_fillers(q_norm)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


# =========================
# 统一字段与保存
# =========================

def _normalize_record(rec: dict) -> dict:
    """统一字段：expected_intent / test_type / tags / domain / case_id / difficulty / type兼容"""
    r = dict(rec)
    # 兼容别名
    if "intent" in r and "expected_intent" not in r:
        r["expected_intent"] = r.pop("intent")
    if "type" in r and "test_type" not in r:
        r["test_type"] = r.pop("type")

    # 填缺省
    r.setdefault("expected_intent", "fallback_intent")
    r.setdefault("test_type", "BASE")
    r.setdefault("tags", [])
    r.setdefault("domain", "general")
    r.setdefault("difficulty", 2)
    if not r.get("case_id"):
        r["case_id"] = f"{r['expected_intent']}-{uuid.uuid4().hex[:8]}"
    return r

def _save_cases(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # parquet（若 pyarrow 缺失会失败，自动回退只存 csv）
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        pass
    df.to_csv(out_path.replace(".parquet", ".csv"), index=False, encoding="utf-8-sig")


# =========================
# 主流程
# =========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML 配置文件（包含 llm/augment 等）")
    p.add_argument("--desc", required=True, help="一句话/短段产品需求描述")
    p.add_argument("--out", required=True, help="输出 parquet 路径（同时导出 CSV）")
    p.add_argument("--total", type=int, default=200, help="期望基础+同义总量（LLM 生成的近似目标）")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg.setdefault("generation", {})["total"] = args.total
    aug_cfg = cfg.get("augment", {}) or {}

    # ============ 1) 基础 + 同义：完全由 LLM 直生 ============
    # 期望返回结构： [{"query": "...", "expected_intent": "...", "test_type":"BASE/SYN", "design_logic":"LLM 直生…", "tags":[...]}]
    base_syn_cases = LG.gen_for_inventory(cfg, {"desc": args.desc}) or []
    base_syn_cases = [_normalize_record(x) for x in base_syn_cases]

    # ============ 2) 增强：NOISE / SLANG / DIALECT / TYPO / CTX / SAFETY ============
    augmented = []

    # 2.1 对每条基础样本做噪声/口语/方言/错写增强
    typo_n    = int(aug_cfg.get("typo_per_base", 0))
    slang_n   = int(aug_cfg.get("slang_per_base", 0))
    dialect_n = int(aug_cfg.get("dialect_per_base", 0))
    noise_n   = int(aug_cfg.get("noise_per_base", 0))

    if any([typo_n, slang_n, dialect_n, noise_n]):
        for rec in base_syn_cases:
            q = rec["query"]
            intent = rec["expected_intent"]

            def _emit(kind, times):
                for _ in range(times):
                    outs = LG.gen_for_description_by_types(cfg, q, type_counts={kind: 1}) or []
                    for nq in outs:
                        augmented.append({
                            "query": nq,
                            "expected_intent": intent,
                            "test_type": kind,
                            "tags": [kind],
                            "design_logic": f"LLM增强：{kind}"
                        })

            _emit("TYPO", typo_n)
            _emit("SLANG", slang_n)
            _emit("DIALECT", dialect_n)
            _emit("NOISE", noise_n)

    # 2.2 上下文（多轮/消歧）
    ctx_needed = int(aug_cfg.get("ctx_per_intent", 0))
    if ctx_needed > 0:
        ctx_cases = LG.gen_for_description_by_types(cfg, base_syn_cases, type_counts={"CTX": ctx_needed}) or []
        for x in ctx_cases:
            x.setdefault("test_type", "CTX")
            x.setdefault("tags", ["CTX"])
            x.setdefault("design_logic", "LLM增强：上下文/多轮/消歧")
        augmented.extend(ctx_cases)

    # 2.3 安全/边界
    safety_min = int(aug_cfg.get("safety_min", 0))
    if safety_min > 0:
        safety_cases = LG.gen_for_description_by_types(cfg, "safety", type_counts={"SAFETY": safety_min}) or []
        for s in safety_cases:
            s.setdefault("test_type", "SAFETY")
            s.setdefault("tags", ["SAFETY"])
            s.setdefault("expected_intent", "拒答")
            s.setdefault("design_logic", "LLM增强：安全/边界对抗")
        augmented.extend(safety_cases)

    # 2.x 合并与字段统一
    all_cases = base_syn_cases + augmented
    all_cases = [_normalize_record(x) for x in all_cases]

    # ============ 3) 归一清洗 + 强去重 ============
    # 去掉“中点”等奇怪符号差异的重复；仅以“句子本身”作为去重键（不考虑标签/类型差异）
    for r in all_cases:
        # 仅做轻量清洗：不改动业务语义
        r["query"] = normalize_query(r.get("query", ""))

    all_cases = dedup_records(all_cases)

    # ============ 4) 保存 ============
    df = pd.DataFrame(all_cases)
    _save_cases(df, args.out)

    # ============ 5) 摘要打印 ============
    vc = df["test_type"].value_counts(dropna=False).to_dict()
    print(json.dumps({
        "saved": args.out,
        "total": int(len(df)),
        "by_test_type": vc,
        "note": "LLM-only; 已做归一清洗+强去重（仅句子维度；差标点视同句）。"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



