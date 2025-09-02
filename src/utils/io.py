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

import sqlite3

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generated_cases (
            case_id TEXT PRIMARY KEY,
            query TEXT,
            test_type TEXT,
            expected_intent TEXT,
            domain TEXT,
            difficulty INTEGER,
            design_logic TEXT,
            tags TEXT,
            context TEXT,
            group_id TEXT,
            step TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(df, db_path):
    conn = sqlite3.connect(db_path)
    df.to_sql('generated_cases', conn, if_exists='append', index=False)
    conn.close()

def rand_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def now_version():
    import datetime as dt
    return dt.datetime.now().strftime('v%Y%m%d')
