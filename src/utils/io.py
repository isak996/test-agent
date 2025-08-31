from pathlib import Path
import pandas as pd
import json, uuid

def ensure_parent(path): Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_table(df, path):
    ensure_parent(path)
    p=str(path)
    if p.endswith('.parquet'):
        try:
            df.to_parquet(p, index=False)
        except Exception:
            df.to_csv(p.replace('.parquet','.csv'), index=False)
            return p.replace('.parquet','.csv')
    else:
        df.to_csv(p, index=False)
    return p

def load_yaml(path):
    import yaml
    return yaml.safe_load(open(path,'r',encoding='utf-8'))

def load_cases(path):
    p=str(path)
    return pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p)

def rand_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def now_version():
    import datetime as dt
    return dt.datetime.now().strftime('v%Y%m%d')
