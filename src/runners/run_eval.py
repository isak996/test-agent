import argparse, time, json, importlib, pandas as pd, requests
from ..utils.io import load_cases, ensure_parent
from ..evaluators.metrics import compute_metrics, save_report

def call_pyfunc(path, query, context=None):
    module, func = path.split(':',1)
    mod = importlib.import_module(module)
    return getattr(mod, func)(query=query, context=context)

def call_api(url, query, context=None, timeout=10.0):
    payload={'query':query};
    if context: payload['context']=context
    r=requests.post(url,json=payload,timeout=timeout); r.raise_for_status(); return r.json()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--api-url', default=None)
    ap.add_argument('--py-func', default='src.utils.demo_nlu:predict_intent')
    ap.add_argument('--report', required=True)
    ap.add_argument('--pred-out', default=None)
    ap.add_argument('--metrics-out', default=None)
    args=ap.parse_args()
    cases=load_cases(args.cases)
    preds=[]
    for _,r in cases.iterrows():
        q=str(r['query']); ctx=r.get('context') if 'context' in r else None
        t0=time.time()
        try:
            res=call_api(args.api_url,q,ctx) if args.api_url else call_pyfunc(args.py_func,q,ctx)
            dt=int((time.time()-t0)*1000)
            preds.append({'case_id':r['case_id'],'intent_pred':res.get('intent',''),'confidence':res.get('confidence',0.0),'topk':json.dumps([x.get('intent',x) if isinstance(x,dict) else x for x in res.get('top_k',res.get('topk',[]))], ensure_ascii=False),'latency_ms':dt,'errors':''})
        except Exception as e:
            preds.append({'case_id':r['case_id'],'intent_pred':'','confidence':0.0,'topk':'[]','latency_ms':0,'errors':str(e)})
    preds_df=pd.DataFrame(preds)
    metrics=compute_metrics(cases, preds_df, k=3)
    ensure_parent(args.report); save_report(metrics, args.report)
    pred_out=args.pred_out or args.report.replace('report.md','predictions.parquet')
    metrics_out=args.metrics_out or args.report.replace('report.md','metrics.json')
    preds_df.to_parquet(pred_out, index=False)
    open(metrics_out,'w',encoding='utf-8').write(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps({'report':args.report,'predictions':pred_out,'metrics':metrics_out,'summary':metrics}, ensure_ascii=False))

if __name__=='__main__': main()
