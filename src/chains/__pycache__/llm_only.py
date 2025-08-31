# src/chains/llm_generators.py
# -*- coding: utf-8 -*-
"""
纯 LLM 生成器：
- 仅按照一句话描述 desc 生成测试查询（BASE/SYN），不跑模板/不插槽。
- 生成后做强清洗：NFKC 归一、去中点“·”与异常标点、去表情/emoji、空白规范。
- 归一文本哈希去重：同句不同标点/空白视为重复。
- 不做“禁止词/领域词”过滤，贴域由提示词严格约束。

对外方法：
- gen_for_description(cfg, desc, total=120) -> List[dict]
- gen_for_inventory(cfg, inv) 兼容旧入口（若 inv 带 desc 则用 desc，否则拼一个概述）

兼容 run_generation 中的调用：
- gen_noise/ gen_ctx/ gen_safety 提供空实现以避免产生“嗯/嘛/·”等噪声。
"""

import re, os, json, uuid, unicodedata, hashlib
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from ..llm_providers.provider import get_llm

# ---------------- 清洗与去重 ----------------

# 要删除的中点/变体与非常见标点（可按需扩展）
_DROP_CHARS = "·•●・．∙‧…~～—_~`^｜|"
# 统一丢掉的中英文标点（为了让“同句不同标点”也能判重复）
_PUNCT = r"""[\u3000\s\.,，。!！?？;；:：、'"“”‘’\(\)\[\]\{\}<>《》【】\-+*=\\/]"""

_EMOJI_RE = re.compile(r"[\U00010000-\U0010FFFF]")  # emoji 等增补平面
_MULTI_SPACE = re.compile(r"\s+")

def _normalize_text(s: str) -> str:
    """NFKC 归一 + 去掉中点/奇怪符号 + 丢常见标点 + 去 emoji + 空白合一"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for ch in _DROP_CHARS:
        s = s.replace(ch, "")
    s = _EMOJI_RE.sub("", s)
    s = re.sub(_PUNCT, " ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s

def _sig(s: str) -> str:
    """基于归一文本的签名；用于强去重"""
    return hashlib.md5(_normalize_text(s).lower().encode("utf-8")).hexdigest()

def _dedup_keep_order(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in lines:
        k = _sig(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out

# ---------------- Prompt 与 LLM 调用 ----------------

def _build_prompt(desc: str, n: int) -> ChatPromptTemplate:
    rules = f"""
你是测试专家。请**只根据下述产品/场景描述**生成中文“测试查询（query）”集合，用于产品测试， NLU/语义理解评测。

必须遵守：
1) 语义必须严格贴合场景描述，不要越域到其他行业/场景。
2) 句子自然口语化，**不要**中点“·”，不要奇怪标点、表情夹杂。
3) 禁止重复或仅标点不同的句子；表达要多样，覆盖“标准表达 + 同义表达”（意思一致说法不同）。
4) 每条 5~25 个汉字左右；不要很长的段落或很短的词语。
5) 仅输出**JSON 数组**（每个元素是字符串），不要任何解释、不要包裹代码块。
6）生成有语气词的query，模拟人真实对话指令query

场景描述：
{desc}

请一次性输出 {n} 条高质量 query（JSON 数组）。
"""
    return ChatPromptTemplate.from_messages([
        ("system", "你是严格的中文测试集生成器，擅长控制多样性、避免重复、贴域生成。"),
        ("user", rules)
    ])

def _invoke_llm_json_list(llm, prompt: ChatPromptTemplate) -> List[str]:
    """调用 LLM，尽力解析为字符串数组；失败时做鲁棒提取。"""
    out = (prompt | llm).invoke({})
    text = out.content if hasattr(out, "content") else str(out)
    # 直读 JSON
    try:
        data = json.loads(text)
        arr = [x for x in data if isinstance(x, str)]
        if arr:
            return arr
    except Exception:
        pass
    # 兜底：粗暴抽引号段
    arr = re.findall(r'"([^"\n]{2,60})"', text)
    return arr

# ---------------- 主入口：一句话 → 结构化用例 ----------------

def gen_for_description(cfg: Dict[str, Any], desc: str, total: int = 120) -> List[Dict[str, Any]]:
    """
    基于“一句话/段落描述”生成测试集（仅 BASE/SYN 两类）。
    - total：目标条数（内部分批生成，做清洗/去重后合并）。
    """
    llm = get_llm(cfg)
    batch = min(60, max(10, total))  # 单批 10~60
    pieces: List[str] = []

    remain = total
    while remain > 0:
        n = min(batch, remain)
        prompt = _build_prompt(desc, n)
        arr = _invoke_llm_json_list(llm, prompt)

        # 基础清洗
        arr = [a.strip() for a in arr if isinstance(a, str)]
        # 去中点“·”等奇符号（生成端已要求，这里再兜一层）
        arr = [re.sub(r"[·•●・．∙‧]", "", a) for a in arr]
        # 空白合一
        arr = [re.sub(r"\s+", " ", a).strip() for a in arr]
        # 长度约束
        arr = [a for a in arr if 4 <= len(a) <= 40]

        pieces.extend(arr)
        remain -= n

    # 强去重（按归一文本签名）
    uniq = _dedup_keep_order(pieces)

    # 拆成 BASE/SYN（前半 BASE，后半 SYN）
    half = max(1, len(uniq) // 2)
    base, syn = uniq[:half], uniq[half:]

    def _rec(q: str, tp: str) -> Dict[str, Any]:
        return {
            "case_id": f"{tp}-{uuid.uuid4().hex[:8]}",
            "query": q,
            "expected_intent": "fallback_intent",   # 如果你后续要细分意图，可再做 LLM 标注链
            "test_type": "BASE" if tp == "BASE" else "SYN",
            "domain": "general",
            "difficulty": 2,
            "design_logic": "LLM 直生（无模板/无插槽）；已做归一清洗+强去重",
            "tags": []
        }

    cases = [_rec(q, "BASE") for q in base] + [_rec(q, "SYN") for q in syn]
    return cases

# -------- 兼容旧入口（inv 里如果自带 desc 就按 desc 走） --------

def gen_for_inventory(cfg: Dict[str, Any], inv: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 若 inv 带 desc，直接用；否则拼一个粗描述（仅为兼容，建议统一走 --desc）
    desc = inv.get("desc")
    if not desc:
        intents = inv.get("intents", [])
        domains = sorted({i.get("domain", "") for i in intents if i.get("domain")})
        title = "、".join(domains) if domains else "通用中文智能系统"
        desc = f"{title} 的中文测试查询集合"
    # 读取配置的默认产量
    total = int(cfg.get("generation", {}).get("total", 120))
    return gen_for_description(cfg, desc, total=total)

# -------- 兼容 run_generation 里可能引用到的函数（返回空） --------

def gen_noise(q: str):
    # 纯 LLM方案：不再人为制造“嗯/嘛/·/口语词”等噪声，以免引入无意义变体
    return {"TYPO": [], "SLANG": [], "DIALECT": [], "NOISE": []}

def gen_ctx(intent_id: str, seed_list):
    # 本迭代不做上下文链；如需可另建 Context 生成器
    return []

def gen_safety():
    # 若 desc 中写了“含安全测试/敏感对抗”，会在主生成里体现；这里不再额外增补
    return []
