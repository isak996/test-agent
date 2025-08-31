# src/llm_providers/provider.py
import os
from typing import Any, Dict

# 使用新版客户端路径：langchain-openai
try:
    from langchain_openai import ChatOpenAI
except Exception:
    # 兼容没装新包的场景，但仍建议 pip install -U langchain-openai
    from langchain_community.chat_models import ChatOpenAI  # type: ignore


def _resolve_llm_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一从 环境变量 > agent.yaml 中解析 LLM 连接参数
    """
    llm_cfg = cfg.get("llm", {}) if cfg else {}
    api_key = os.getenv("LLM_API_KEY") or llm_cfg.get("api_key")
    base_url = os.getenv("LLM_BASE_URL") or llm_cfg.get("base_url")
    model = os.getenv("LLM_MODEL") or llm_cfg.get("model")
    temperature = float(llm_cfg.get("temperature", 0.7))
    max_tokens = int(llm_cfg.get("max_tokens", 1024))

    if not (api_key and base_url and model):
        raise RuntimeError(
            "LLM 配置缺失：请设置环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL，"
            "或在 configs/agent.yaml 的 llm 节点下提供 api_key/base_url/model。"
        )
    return dict(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm(cfg: dict, override: dict | None = None):
    """
    支持运行时覆盖 provider / base_url / api_key / model / temperature / max_tokens
    优先级：override > cfg['llm'] > 环境变量
    """
    o = override or {}
    c = (cfg or {}).get("llm", {})

    provider     = o.get("provider")     or c.get("provider")     or "deepseek"
    base_url     = o.get("base_url")     or os.getenv("LLM_BASE_URL") or c.get("base_url")
    api_key      = o.get("api_key")      or os.getenv("LLM_API_KEY")  or c.get("api_key")
    model        = o.get("model")        or os.getenv("LLM_MODEL")    or c.get("model")
    temperature  = o.get("temperature")  or c.get("temperature", 0.7)
    max_tokens   = o.get("max_tokens")   or c.get("max_tokens", 1024)

    # 走 OpenAI 兼容接口（火山/豆包/DeepSeek 网关都支持）
    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    return llm


# 向后兼容：部分模块还引用 get_llm_from_env
def get_llm_from_env():
    """
    仅从环境变量读取（不依赖 agent.yaml），保持与旧代码兼容。
    """
    dummy_cfg = {
        "llm": {
            "api_key": os.getenv("LLM_API_KEY"),
            "base_url": os.getenv("LLM_BASE_URL"),
            "model": os.getenv("LLM_MODEL"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        }
    }
    return get_llm(dummy_cfg)

# src/llm_providers/provider.py
# -*- coding: utf-8 -*-
from langchain_openai import ChatOpenAI  # 你当前环境里可用
# from langchain_community.chat_models import ChatOpenAI  # 若上面不可用，换这一行


