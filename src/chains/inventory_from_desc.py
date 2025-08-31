# src/chains/inventory_from_desc.py
import json
from src.llm_providers.provider import get_llm

PROMPT_TEMPLATE = """你是一个车载语音助手测试专家。
根据以下产品需求描述，提取出所有【意图】和【槽位】：
需求描述: {desc}

请输出 JSON，格式:
{
  "intents": [
    {"id": "intent_name", "domain": "导航/多媒体/车控/闲聊/安全", "templates": ["示例1","示例2"]},
    ...
  ],
  "slots": {
    "artist": ["周杰伦","林俊杰"],
    "poi": ["加油站","医院"],
    "city": ["上海","北京"]
  }
}"""

def inventory_from_desc(cfg, desc: str):
    llm = get_llm(cfg)
    prompt = PROMPT_TEMPLATE.format(desc=desc)
    resp = llm.invoke(prompt)
    try:
        inv = json.loads(resp.content if hasattr(resp, "content") else resp)
    except Exception as e:
        raise ValueError(f"LLM返回无法解析为JSON: {resp}") from e
    return inv
