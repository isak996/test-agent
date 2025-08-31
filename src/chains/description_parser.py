# src/chains/description_parser.py
# -*- coding: utf-8 -*-
import sys, json, re, unicodedata
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from ..llm_providers.provider import get_llm

def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()

def _prompt_for_taxonomy(desc: str, min_domains: int = 4, max_domains: int = 8, intents_per_domain: int = 6):
    # 注意：示例 JSON 的所有花括号都用成了 {{ }}，避免被 ChatPromptTemplate 当成变量
    text = f"""
你是出色的产品NLU专家。请把下面的产品/场景描述拆解成**功能域(domain)**与**意图(intent)**的图谱。
必须**只输出 JSON**，结构如下：
{{
  "domains": [
    {{"name":"导航","intents":["路线规划","周边搜索","停车推荐"]}},
    ...
  ]
}}
要求：
- 每个 domain.name 是中文短语（2~8字）；每个 intents 为 3~{intents_per_domain} 个简短意图名；
- 总 domain 数量 ≥ {min_domains} 且 ≤ {max_domains}；
- 覆盖全面、紧贴描述；不要解释文字，不要代码块。
场景描述：
{desc}
## 生成domain思考提示： 
- 用户在此场景下会有哪些需求？ 
- 不同用户角色可能有哪些差异化需求？ 
- 该产品的生命周期各阶段需要什么功能？ 
- 什么功能是必需的，什么是锦上添花的？ 
- 最基础、最核心的功能域 - 信息娱乐与环境控制 
- 通讯与社交 - 生活服务与生产力 
- 安全、安防与辅助 
- 个性化与主动服务 
- 未来概念与前沿功能作为一个无所不包、无时不在、主动贴心的超级智能体，应该覆盖的所有domain
"""
    # 把上面示例里的 { 和 } 转成 {{ 和 }}
    text = (
        text
        .replace("{", "{{")
        .replace("}", "}}")
        # 但把真正的变量 {desc} 再换回来
        .replace("{{desc}}", "{desc}")
        .replace("{{min_domains}}", str(min_domains)) \
        .replace("{{max_domains}}", str(max_domains)) \
        .replace("{{intents_per_domain}}", str(intents_per_domain))
    )

    return ChatPromptTemplate.from_messages([
        ("system", "你是严谨的中文功能建模专家，擅长将产品描述拆成 domain/intent 图谱。只输出 JSON。"),
        ("user", text)
    ])

_CODE_FENCE_BLOCK = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", re.MULTILINE)
_CODE_FENCE_EDGE  = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)

def _strip_code_fence(text: str) -> str:
    t = text.strip()
    m = _CODE_FENCE_BLOCK.search(t)
    if m:
        return m.group(1).strip()
    return _CODE_FENCE_EDGE.sub("", t).strip()

def _extract_json_dict(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    s = cleaned
    start = s.find("{")
    while start != -1:
        lvl = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                lvl += 1
            elif c == "}":
                lvl -= 1
                if lvl == 0:
                    candidate = s[start:i+1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict):
                            return data
                    except Exception:
                        break
        start = s.find("{", start + 1)

    raise ValueError("No JSON object found")

def parse_domains_intents(cfg: Dict[str, Any], desc: str) -> Dict[str, Any]:
    desc = _nfkc(desc)
    llm = get_llm(cfg)
    # 从 cfg 中获取 taxonomy 配置
    taxonomy_cfg = cfg.get("taxonomy", {})
    min_domains = taxonomy_cfg.get("min_domains", 4) # 默认值
    max_domains = taxonomy_cfg.get("max_domains", 8) # 默认值
    intents_per_domain = taxonomy_cfg.get("intents_per_domain", 6) # 默认值

    prompt = _prompt_for_taxonomy(desc, min_domains, max_domains, intents_per_domain)
    resp = (prompt | llm).invoke({})
    content = getattr(resp, "content", str(resp))

    try:
        data = _extract_json_dict(content)
    except Exception as e:
        print(f"[warn] LLM response parse failed: {e}, content:\n{content}", file=sys.stderr)
        return {"desc": desc, "domains": []}

    domains_raw = data.get("domains") or []
    out: List[Dict[str, Any]] = []
    for d in domains_raw:
        if not isinstance(d, dict):
            continue
        name = _nfkc(d.get("name", "")) or "general"
        intents = [ _nfkc(x) for x in (d.get("intents") or []) if _nfkc(x) ]
        intents = intents[:intents_per_domain]
        if intents:
            out.append({"name": name, "intents": intents})

    out = out[:max_domains]
    return {"desc": desc, "domains": out}
