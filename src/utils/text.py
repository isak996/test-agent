
# -*- coding: utf-8 -*-
"""
Text augmentation helpers for the test agent.
This module provides lightweight, dependency-free perturbations:
- add_typo: common Chinese char swaps / homoglyphs
- add_slang: colloquial fillers/particles
- add_dialect: simple regional word variants
- add_noise: wrappers with hesitations / redundant words
"""
import random
import re

# Common replacements (very small demo set; extend as needed)
TYPO_MAP = {
    "音乐": ["音樂", "音玥", "⾳乐"],
    "导航": ["到航", "導航"],
    "加油站": ["加油栈", "加油點"],
    "电影院": ["電影院", "影院"],
    "银行": ["銀⾏", "銀行"],
    "周杰伦": ["周傑倫", "周杰倫"],
    "陈奕迅": ["陳奕迅", "陳奕⾨"],
}

SLANG_FILLERS = [
    "那个啥", "就是", "然后", "拜托啦", "劳驾", "呃", "嗯", "诶", "麻烦你", "请问一下"
]

DIALECT_REPLACEMENTS = [
    # (pattern, replacement)
    ("这里", "这块"),           # 北方口语
    ("那里", "那块"),
    ("有点", "有点儿"),
    ("帮我", "给我整"),         # 东北口语风
    ("修车厂", "修理铺子"),
    ("便利店", "小卖部"),
]

NOISE_PARTICLES = ["嘛", "呗", "啦", "呀", "哈", "咯"]
NOISE_FILLERS = ["那个啥…", "嗯…", "诶…", "就是…", "拜托…", "请问…"]

def _safe_replace_once(s, src, tgt):
    if src in s:
        return s.replace(src, tgt, 1)
    return s

def add_typo(s: str) -> str:
    # random char-level replacement based on TYPO_MAP
    candidates = [k for k in TYPO_MAP.keys() if k in s]
    if candidates:
        k = random.choice(candidates)
        v = random.choice(TYPO_MAP[k])
        return _safe_replace_once(s, k, v)
    # fallback: insert a homoglyph-like dot
    if len(s) > 2:
        pos = random.randrange(1, len(s))
        return s[:pos] + "·" + s[pos:]
    return s

def add_slang(s: str) -> str:
    filler = random.choice(SLANG_FILLERS)
    if random.random() < 0.5:
        return f"{filler}…{s}"
    return f"{s}，{filler}"

def add_dialect(s: str) -> str:
    if DIALECT_REPLACEMENTS:
        pat, rep = random.choice(DIALECT_REPLACEMENTS)
        return re.sub(pat, rep, s, count=1)
    return s

def add_noise(s: str) -> str:
    """Add light-weight noise around a sentence.
    - prefix a hesitation filler OR
    - suffix a particle
    """
    choice = random.random()
    if choice < 0.5:
        filler = random.choice(NOISE_FILLERS)
        return f"{filler}{s}"
    else:
        particle = random.choice(NOISE_PARTICLES)
        return f"{s}{particle}"
