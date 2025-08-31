# src/runners/run_generation.py
# -*- coding: utf-8 -*-
import argparse, os, json
import pandas as pd
import yaml

from ..chains.description_parser import parse_domains_intents
from ..chains import llm_generators as LG

DEFAULT_ALLOC = {"BASE": 10, "SYN": 10, "NOISE": 10, "SLANG": 10,
                 "DIALECT": 10, "TYPO": 10, "CTX": 10, "SAFETY": 10}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--desc', required=True, help='一句话/产品描述，自动解析 domain 并按类型生成')
    p.add_argument('--out', required=True)
    p.add_argument('--domains-max', type=int, default=8)
    p.add_argument('--intents-per-domain', type=int, default=6)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))

    # 读取 8 类配额；如未配置则用默认
    alloc = (cfg.get('generation', {}) or {}).get('allocation', DEFAULT_ALLOC)
    # 只保留允许的类型（避免拼错）
    alloc = {k.upper(): int(v) for k, v in alloc.items()
             if k.upper() in {"BASE","SYN","NOISE","SLANG","DIALECT","TYPO","CTX","SAFETY"} and int(v) > 0}
    if not alloc:
        alloc = {"BASE": 10, "SYN": 10, "NOISE": 10, "SLANG": 10,
                 "DIALECT": 10, "TYPO": 10, "CTX": 10, "SAFETY": 10}  # 兜底
    print("[info] type allocation:", alloc)

    # 先用 LLM 解析 domains / intents
    taxonomy = parse_domains_intents(cfg, args.desc, max_domains=args.domains_max, intents_per_domain=args.intents_per_domain)
    domains = taxonomy.get("domains") or []
    if not domains:
        print("[warn] no domains parsed, fallback to single 'general'")
        domains = [{"name": "general", "intents": []}]

    print("[info] parsed domains:", [d["name"] for d in domains])

    # 按每个 domain 调用一次 LLM 生成（逐类型）
    all_cases = []
    for d in domains:
        dname = d.get("name") or "general"
        # 构造一个域内描述，帮助模型保持贴域
        sub_desc = f"{args.desc} —— 功能域：{dname}。覆盖意图：{('、'.join(d.get('intents') or [])) or '该域常见意图'}。"
        cases = LG.gen_for_description_by_types(cfg, sub_desc, alloc)  # 关键：逐类型生成
        # 给该 batch 打上 domain 字段（LLM 也可能返回 domain，这里以我们的为准）
        for c in cases:
            c["domain"] = dname
        print(f"[info] domain={dname} generated={len(cases)}")
        all_cases.extend(cases)

    # 保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(all_cases)
    # 兼容没有 pyarrow 的环境
    try:
        df.to_parquet(args.out, index=False)
    except Exception:
        pass
    df.to_csv(args.out.replace('.parquet', '.csv'), index=False, encoding='utf-8-sig')

    # 汇总信息
    by_type = df['test_type'].value_counts().to_dict() if not df.empty else {}
    by_domain = df['domain'].value_counts().to_dict() if not df.empty else {}
    print(json.dumps({"saved": args.out, "total": len(df), "by_type": by_type, "by_domain": by_domain}, ensure_ascii=False))

if __name__ == '__main__':
    main()
