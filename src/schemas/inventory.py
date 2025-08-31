# src/schemas/inventory.py
import yaml

REQUIRED_TOP_LEVEL = ["intents"]

def load_inventory_yaml(path: str) -> dict:
    """
    读取 intents.yaml 并做最小规范化：
    - 必须包含 'intents'
    - slots 可选，若缺失则置空 {}
    - 每个 intent 至少包含 id；缺失 templates 则补 []
    - 允许 domain/description 可选
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            raise ValueError(f"Inventory YAML missing required key: '{key}'")

    data.setdefault("slots", {})
    intents = data.get("intents") or []
    if not isinstance(intents, list) or not intents:
        raise ValueError("Inventory 'intents' must be a non-empty list")

    normalized = []
    for it in intents:
        if not isinstance(it, dict) or "id" not in it:
            raise ValueError("Each intent must be an object with at least an 'id' field")
        it.setdefault("templates", [])
        it.setdefault("domain", "")
        it.setdefault("description", "")
        normalized.append(it)

    data["intents"] = normalized
    return data
