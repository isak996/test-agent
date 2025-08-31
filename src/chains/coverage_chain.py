# src/chains/coverage_chain.py
# -*- coding: utf-8 -*-
import math
from typing import Dict, Any
import pandas as pd

ALL_TYPES = ["BASE","SYN","NOISE","SLANG","DIALECT","TYPO","CTX","SAFETY"]

def _cfg_generation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    gen = (cfg or {}).get("generation", {})
    total = int(gen.get("total", 120))
    ratios = gen.get("ratios", {"BASE":0.5,"SYN":0.5})
    # 只保留已知类型并归一化
    ratios = {k: float(v) for k,v in ratios.items() if k in ALL_TYPES}
    s = sum(ratios.values()) or 1.0
    ratios = {k: v/s for k,v in ratios.items()}
    return {"total": total, "ratios": ratios}

def audit_coverage(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    g = _cfg_generation(cfg)
    want_total, ratios = g["total"], g["ratios"]

    by_type_now = df["test_type"].value_counts().to_dict()
    now_total = int(len(df))

    need_by_type: Dict[str,int] = {}
    for t in ALL_TYPES:
        desired = int(math.ceil(want_total * ratios.get(t, 0.0)))
        have = int(by_type_now.get(t, 0))
        need_by_type[t] = max(0, desired - have)

    # 如果总量不够且各类都达标（极少见），把余量补到 BASE
    delta_total = max(0, want_total - now_total)
    if delta_total > 0 and all(v == 0 for v in need_by_type.values()):
        need_by_type["BASE"] = delta_total

    return {
        "now": {"total": now_total, "by_type": by_type_now},
        "target": {"total": want_total, "ratios": ratios},
        "need_by_type": {k:int(v) for k,v in need_by_type.items() if v>0},
    }
