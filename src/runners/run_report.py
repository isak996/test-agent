import argparse, json
from ..utils.io import load_cases
from ..evaluators.metrics import compute_metrics, save_report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cases', required=True); ap.add_argument('--preds'); ap.add_argument('--report', required=True); args=ap.parse_args()
    cases=load_cases(args.cases); preds=load_cases(args.preds) if args.preds else cases[['case_id']].assign(intent_pred='',topk='[]',confidence=0,latency_ms=0,errors='')
    metrics=compute_metrics(cases, preds, k=3); save_report(metrics, args.report); print(json.dumps(metrics, ensure_ascii=False))

if __name__=='__main__': main()
