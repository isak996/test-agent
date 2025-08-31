# webapp/app.py
# -*- coding: utf-8 -*-
import os, io, time, sys, pathlib, json, tempfile, math
import pandas as pd
import streamlit as st

# 让 Python 能找到你的 src 包（假设 webapp/ 与 src/ 同级）
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ——严格使用项目里的实现，不做页面级兜底/重试——
from src.chains.description_parser import parse_domains_intents
from src.chains.llm_generators import (
    _ALLOWED_TYPES,              # 允许的类型集合
    gen_for_description_by_types # 批量生成函数
)
from src.llm_providers.provider import get_llm  # 使用你项目里的 provider

# ============== 页面基础信息 ==============
st.set_page_config(page_title="NLU 测试集生成器", page_icon="🧪", layout="wide")
st.title("🧪 测试集生成器")
st.caption("输入一句话/产品描述 →  生成多样化中文测试 Query → 下载 CSV/Parquet")

# ============== 侧边栏参数区 ==============
with st.sidebar:
    st.header("⚙️ 基本参数")

    desc = st.text_area(
        "产品/场景一句话描述",
        value="车载语音智能助手",
        height=100,
        placeholder="例如：智能家居语音助手，控制灯光/空调/扫地机器人，含闲聊/上下文/安全兜底",
        key="inp_desc",
    )

    total = st.number_input(
        "目标总量（条）",
        min_value=20, max_value=10000, value=256, step=16,
        key="inp_total",
    )

    # 是否将 total 按域均分（强烈建议打开，避免“总量×域数”的调用放大）
    even_by_domain = st.checkbox(
        "将总量按域均分（推荐）",
        value=True,
        key="chk_even_by_domain"
    )

    st.markdown("**类型配额（比例 0~1 或具体整数，自动归一到总量）**")
    default_alloc = {
        "BASE": 0.20, "SYN": 0.20,
        "NOISE": 0.10, "SLANG": 0.10, "DIALECT": 0.10, "TYPO": 0.10,
        "CTX": 0.10, "SAFETY": 0.10
    }
    alloc_inputs = {}
    for i, t in enumerate(["BASE","SYN","NOISE","SLANG","DIALECT","TYPO","CTX","SAFETY"]):
        alloc_inputs[t] = st.text_input(t, value=str(default_alloc[t]), key=f"alloc_{t}_{i}")

    st.divider()
    st.header("🤖 模型与采样")

    MODEL_OPTIONS = {
        "满血 DeepSeek-R1 (推理)": {
            "provider": "deepseek",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "ep-20250212182831-dgcw4",
        },
        "豆包 多模态 Pro": {
            "provider": "doubao",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "ep-20250508134742-6njfp",
        },
        "豆包 文本 Pro": {
            "provider": "doubao",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "ep-20250217002939-lfc9n",
        },
    }

    model_choice = st.selectbox(
        "选择模型",
        list(MODEL_OPTIONS.keys()),
        index=0,
        key="sel_model",
    )

    # API Key 用环境变量或这里输入（优先这里）
    api_key_input = st.text_input(
        "API Key",
        type="password",
        value=os.getenv("LLM_API_KEY", ""),
        key="inp_api_key",
        help="建议在部署环境使用 Secrets；本地可 export LLM_API_KEY。"
    )

    temperature = st.slider(
        "temperature（采样多样性）",
        min_value=0.0, max_value=1.5,
        value=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        step=0.05,
        key="sld_temp",
    )
    max_tokens = st.number_input(
        "max_tokens",
        min_value=256, max_value=8192,
        value=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        step=128,
        key="inp_max_tokens",
    )

    # 高级：显示每次 LLM 调用的详细日志
    debug_show_logs = st.checkbox("显示详细实时日志", value=True, key="chk_debug_logs")

    run_btn = st.button("🚀 生成测试集", type="primary", key="btn_run")

# ============== 工具函数 ==============
def resolve_alloc(total_count: int, raw: dict):
    """
    支持比例(0.x)或具体整数；自动归一到 total_count。
    仅保留 _ALLOWED_TYPES，且 >0 的条目。
    """
    # 判断是否按比例
    is_ratio = True
    try:
        for v in raw.values():
            if float(v) > 1:
                is_ratio = False
                break
    except Exception:
        is_ratio = False

    out = {}
    if is_ratio:
        for k, v in raw.items():
            out[k] = int(round(float(v) * total_count))
        # 四舍五入差额补到 BASE
        delta = total_count - sum(out.values())
        out["BASE"] = out.get("BASE", 0) + delta
    else:
        # 具体整数
        for k, v in raw.items():
            try:
                out[k] = int(float(v))
            except Exception:
                out[k] = 0
        s = sum(out.values())
        if s != total_count and s > 0:
            # 比例缩放到 total_count
            out = {k: int(round(v * total_count / s)) for k, v in out.items()}
            delta = total_count - sum(out.values())
            out["BASE"] = out.get("BASE", 0) + delta

    # 只保留允许的类型 & >0
    out = {k: v for k, v in out.items() if k in _ALLOWED_TYPES and v > 0}
    return out

def build_cfg(model_choice: str, temperature: float, max_tokens: int, total: int, api_key: str):
    """
    从 UI 构造 cfg，显式覆盖 llm 信息（项目里的 provider 会先读 cfg，再读环境变量）。
    """
    opt = MODEL_OPTIONS[model_choice]
    cfg = {
        "llm": {
            "provider": opt["provider"],  # deepseek / doubao（内部统一走 OpenAI 兼容）
            "model": opt["model"],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "base_url": opt["base_url"],
        },
        "generation": {
            "total": int(total),
        },
        # 给下游一个运行时 override（如你的 provider.py 支持，会直接读这里）
        "_override": {
            "provider": opt["provider"],
            "model": opt["model"],
            "base_url": opt["base_url"],
            "api_key": api_key.strip(),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
    }

    # 同时设置环境变量（保证 provider 能读到）
    os.environ["LLM_BASE_URL"] = opt["base_url"]
    os.environ["LLM_MODEL"] = opt["model"]
    if api_key.strip():
        os.environ["LLM_API_KEY"] = api_key.strip()

    return cfg

def log_init():
    if "live_logs" not in st.session_state:
        st.session_state["live_logs"] = []
def log_line(msg: str):
    st.session_state["live_logs"].append(msg)

# ============== 主流程 ==============
if run_btn:
    if not desc.strip():
        st.error("请先填写产品/场景描述")
        st.stop()

    # 计算类型配额（全局）
    type_counts_global = resolve_alloc(int(total), alloc_inputs)
    if not type_counts_global:
        st.error("请至少为一种类型分配条数/比例")
        st.stop()

    # 构建 cfg（显式覆盖到你项目的 provider）
    cfg = build_cfg(model_choice, temperature, max_tokens, int(total), api_key_input)

    # === 1) 域解析（严格调用你项目里的 parse_domains_intents）===
    st.subheader("🧭 域解析")
    domain_status = st.empty()
    t0 = time.time()
    taxonomy = parse_domains_intents(cfg, desc)
    domains = taxonomy.get("domains", [])
    domain_names = [d.get("name", "general") for d in domains] or ["general"]
    domain_status.success(f"域解析完成：{len(domain_names)} 个 → {', '.join(domain_names)}（用时 {time.time()-t0:.1f}s）")
    with st.expander("查看 Domain 解析 JSON", expanded=False):
        st.json(taxonomy)

    # === 2) 计算“每个域”的类型配额 ===
    if even_by_domain:
        # 把总量按域均分，再按类型拆分
        per_domain_total = max(1, int(round(total / len(domain_names))))
        type_counts = resolve_alloc(per_domain_total, alloc_inputs)
        st.caption(f"💡 已启用“按域均分”：每个域目标≈ {per_domain_total} 条；每域类型配额={type_counts}")
    else:
        # 为每个域使用“全局配额”（会乘以域数）
        type_counts = type_counts_global
        st.caption(f"⚠️ 未启用“按域均分”：每个域都使用全局配额；实际总调用≈ 域数×类型数")

    # === 3) 按域 & 类型生成（实时进度 + 详细日志）===
    st.subheader("🛠️ 用例生成")
    progress_overall = st.progress(0, text="准备中…")
    per_domain_status = st.empty()
    per_type_status = st.empty()
    log_init()
    if debug_show_logs:
        log_area = st.empty()

    call_metrics = []  # 记录每个域/类型调用的指标

    all_rows = []
    total_domains = len(domain_names)
    total_types_per_domain = len(type_counts)
    total_calls = total_domains # Each domain will now make one call to gen_for_description_by_types
    call_done = 0

    for i, d in enumerate(domain_names, start=1):
        per_domain_status.info(f"域 [{i}/{total_domains}]：{d}")
        d_desc = f"{desc}（功能域：{d}）"

        t_start = time.perf_counter()
        per_type_status.info(f"  ↳ 正在为域 {d} 生成所有类型…")

        # ——关键：调用 gen_for_description_by_types 获得所有类型级统计——
        # 假设 gen_for_description_by_types 返回的 rows 已经包含了 domain 信息
        rows = gen_for_description_by_types(cfg, d_desc, type_counts)
        all_rows.extend(rows)

        t_used = time.perf_counter() - t_start
        got = len(rows)
        tps = (got / t_used) if t_used > 0 else 0.0
        call_done += 1
        progress_overall.progress(
            min(100, int(call_done * 100 / max(1, total_calls))),
            text=f"已完成 {call_done}/{total_calls} 次域调用"
        )

        # Simplified metrics for domain-level call
        call_metrics.append({
            "domain": d,
            "need_total": sum(type_counts.values()),
            "got_total": got,
            "time_sec": round(t_used, 2),
            "tps": round(tps, 2),
        })

        if debug_show_logs:
            log_line(f"[{d}] 目标 {sum(type_counts.values())} → 实得 {got}；耗时 {t_used:.2f}s，吞吐 {tps:.2f} q/s")
            log_area.code("\n".join(st.session_state["live_logs"]), language="text")

        per_domain_status.success(f"域 {d} 完成。")

    per_type_status.empty()
    progress_overall.progress(100, text="全部完成")

    # === 4) 汇总展示 ===
    st.success(f"✅ 生成完成，共 {len(all_rows)} 条")
    if not all_rows:
        st.error("没有生成到样本，请检查 Key/网络/模型。")
        st.stop()

    df = pd.DataFrame(all_rows)

    # === 5) 统计面板 ===
    st.subheader("📊 统计")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("总条数", len(df))
    with c2:
        st.metric("域数量", df["domain"].nunique())
    with c3:
        st.metric("类型数量", df["test_type"].nunique())
    with c4:
        st.metric("总调用次数", total_calls)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**类型分布**")
        st.dataframe(
            df["test_type"].value_counts().rename_axis("test_type").reset_index(name="count"),
            use_container_width=True
        )
    with col2:
        st.markdown("**Domain 分布**")
        st.dataframe(
            df["domain"].fillna("general").value_counts().rename_axis("domain").reset_index(name="count"),
            use_container_width=True
        )

    st.markdown("**调用明细（域 × 类型）**")
    df_calls = pd.DataFrame(call_metrics)
    st.dataframe(df_calls, use_container_width=True)

    st.subheader("🔎 预览（Top 100）")
    st.dataframe(df.head(100), use_container_width=True, height=420)

    # === 6) 下载区 ===
    st.subheader("⬇️ 下载")
    # CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
    st.download_button("下载 CSV", data=csv_buf.getvalue(), file_name="testcases.csv", mime="text/csv", key="dl_csv")

    # Parquet（需要 pyarrow）
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmpf:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, tmpf.name)
            tmp_path = tmpf.name
        with open(tmp_path, "rb") as f:
            st.download_button("下载 Parquet", data=f.read(), file_name="testcases.parquet",
                               mime="application/octet-stream", key="dl_parquet")
        os.remove(tmp_path)
    except Exception:
        st.info("如需 Parquet 下载，请在 requirements.txt 中添加 `pyarrow`。")
