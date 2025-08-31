import pandas as pd, json
SAFE={'拒答','不支持','安全拦截','闲聊'}

def compute_metrics(cases, preds, k=3):
    df=cases.merge(preds,on='case_id',how='left',suffixes=('','_preds'))
    total=len(df)
    df['topk_hit']=df.apply(lambda r: (r['expected_intent'] in (json.loads(r['topk']) if isinstance(r['topk'],str) else (r['topk'] or []))[:k]), axis=1)
    top1=(df['expected_intent']==df['intent_pred']).mean()
    topk=df['topk_hit'].mean()
    def acc_where(mask):
        sub=df[mask]
        return None if len(sub)==0 else (sub['expected_intent']==sub['intent_pred']).mean()
    acc_base=acc_where(df['test_type']=='BASE')
    acc_noise=acc_where(df['test_type'].isin(['TYPO','SLANG','DIALECT','NOISE']))
    robust_drop=None if (acc_base is None or acc_noise is None) else max(0.0, acc_base-acc_noise)
    return {'total':int(total),'accuracy_top1':float(top1) if pd.notna(top1) else None,'topk_coverage':float(topk) if pd.notna(topk) else None,'base_accuracy':None if acc_base is None else float(acc_base),'noisy_accuracy':None if acc_noise is None else float(acc_noise),'robustness_drop':None if robust_drop is None else float(robust_drop)}

def save_report(metrics, path_md):
    def pct(x): return '-' if x is None else f"{x*100:.2f}%"
    lines=['# 报告','',f"- 用例总数：{metrics['total']}",f"- Top-1 准确率：{pct(metrics['accuracy_top1'])}",f"- Top-K 覆盖率：{pct(metrics['topk_coverage'])}",f"- 基础场景准确率：{pct(metrics['base_accuracy'])}",f"- 噪声场景准确率：{pct(metrics['noisy_accuracy'])}",f"- 鲁棒性降幅：{pct(metrics['robustness_drop'])}"]
    Path=__import__('pathlib').Path
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)
    open(path_md,'w',encoding='utf-8').write('\n'.join(lines))
