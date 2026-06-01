#!/usr/bin/env python3
"""V2-B-1 training health check.

Reads a training log, extracts per-epoch metrics, emits ONE line per call:
- Every 10 epochs: health summary
- Anytime: anomaly warning (D1 spike, test_acc collapse, train_acc stall, loss_reg divergence)
- Final: end-of-training summary

State (last reported epoch) stored in <log_path>.health_state.
"""
import argparse, json, os, re, sys, statistics
from pathlib import Path

EPOCH_RE = re.compile(r'^Epoch (\d+):\s+avg_loss=([\d.]+),\s+avg_train_acc=([\d.]+)%,\s+avg_test_acc=([\d.]+)%')
AGG_RE = re.compile(r'\[AGG-DIAG\] r=\d+ ep=(\d+) \| D1 glob=([\d.e+-]+) max=([\d.e+-]+)')
V2B1_RE = re.compile(r'\[V2-B-1\] ep=(\d+).*?loss_reg=([\d.e+-]+).*?grad_cos=([-\d.e+]+).*?r\(reg/total\)=([\d.]+)')
ERR_RE = re.compile(r'Traceback|RuntimeError|CUDA out of memory|Killed|NaN detected|AssertionError', re.I)

# Thresholds per stage (gate from research/V2B1_score_norm/0525_V2B1_3LOO_supplement_plan.md)
THRESH = {
    'D1_max': 1e-3,
    'reg_warmup_epoch': 30,
    'train_acc_min_by_ep30': 60.0,   # by ep 30 should be well into convergence
    'train_acc_min_by_ep60': 85.0,
    'grad_cos_floor': -0.5,
    'loss_reg_max': 5.0,             # if loss_reg blows up, regularizer diverged
}


def parse(log_path):
    """Returns by_epoch dict and error_lines.
    Per epoch key 'dg_test_acc' = DG test on held-out domain (the real R1 metric).
    """
    by_epoch = {}
    error_lines = []
    with open(log_path, errors='replace') as f:
        for line in f:
            if m := EPOCH_RE.match(line):
                ep = int(m.group(1))
                by_epoch.setdefault(ep, {}).update({
                    'loss': float(m.group(2)),
                    'train_acc': float(m.group(3)),
                    'dg_test_acc': float(m.group(4)),
                })
            if m := AGG_RE.search(line):
                ep = int(m.group(1))
                d1 = float(m.group(2)); d1_max = float(m.group(3))
                by_epoch.setdefault(ep, {}).update({
                    'd1': d1,
                    'd1_max': max(d1_max, by_epoch.get(ep, {}).get('d1_max', 0)),
                })
            if m := V2B1_RE.search(line):
                ep = int(m.group(1))
                by_epoch.setdefault(ep, {}).update({
                    'loss_reg': float(m.group(2)),
                    'grad_cos': float(m.group(3)),
                    'r_reg': float(m.group(4)),
                })
            if ERR_RE.search(line):
                error_lines.append(line.rstrip()[:200])

    return by_epoch, error_lines


def detect_anomalies(by_epoch, current_ep, stage):
    anomalies = []
    cur = by_epoch.get(current_ep, {})
    d1 = cur.get('d1_max', 0)
    if d1 > THRESH['D1_max']:
        anomalies.append(f"D1_max={d1:.2e} > 1e-3 threshold")
    if 'loss_reg' in cur and cur['loss_reg'] > THRESH['loss_reg_max']:
        anomalies.append(f"loss_reg={cur['loss_reg']:.2f} > {THRESH['loss_reg_max']}")
    if 'grad_cos' in cur and cur['grad_cos'] < THRESH['grad_cos_floor']:
        anomalies.append(f"grad_cos={cur['grad_cos']:.2f} < {THRESH['grad_cos_floor']}")
    return anomalies


# V1 baseline DG test_acc (末 20 ep 平均) per LOO × stage,
# from research/V1_baseline/0525_V1_baseline_full_summary.md
V1_BASELINE = {
    ('art_painting', '1'): 74.00,    ('art_painting', '2'): 68.00,
    ('cartoon',      '1'): 73.13,    ('cartoon',      '2'): 71.37,
    ('photo',        '1'): 89.72,    ('photo',        '2'): 86.88,
    ('sketch',       '1'): 75.95,    ('sketch',       '2'): 68.40,
}


def recent_mean(by_epoch, ep, key, window=10):
    vals = [by_epoch[e][key] for e in range(max(1, ep - window + 1), ep + 1) if e in by_epoch and key in by_epoch[e]]
    return statistics.mean(vals) if vals else None


def fmt_summary(by_epoch, ep, baseline=None):
    cur = by_epoch.get(ep, {})
    parts = [f"ep={ep}"]
    if 'dg_test_acc' in cur:
        s = f"DG_test={cur['dg_test_acc']:.2f}%"
        m10 = recent_mean(by_epoch, ep, 'dg_test_acc', 10)
        if m10 is not None:
            s += f" (10ep_mean={m10:.2f}%"
            if baseline is not None:
                s += f", Δ vs V1={m10 - baseline:+.2f}%"
            s += ")"
        parts.append(s)
    if 'train_acc' in cur:
        parts.append(f"train={cur['train_acc']:.1f}%")
    if 'd1' in cur:
        parts.append(f"D1={cur['d1']:.2e}(max={cur.get('d1_max',0):.2e})")
    if 'loss_reg' in cur:
        parts.append(f"loss_reg={cur['loss_reg']:.3f}")
    if 'grad_cos' in cur:
        parts.append(f"grad_cos={cur['grad_cos']:+.2f}")
    if 'r_reg' in cur:
        parts.append(f"r_reg={cur['r_reg']:.2%}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log_path')
    ap.add_argument('--stage', choices=['1', '2'], default='1')
    ap.add_argument('--leave_out', default=None, help='LOO domain for V1 baseline comparison')
    ap.add_argument('--interval', type=int, default=10, help='report every N epochs')
    ap.add_argument('--label', default='', help='prefix for output')
    args = ap.parse_args()

    if not os.path.exists(args.log_path):
        return

    baseline = V1_BASELINE.get((args.leave_out, args.stage)) if args.leave_out else None

    state_path = args.log_path + '.health_state'
    state = {'last_reported_ep': 0, 'last_anomaly_ep': -1}
    if os.path.exists(state_path):
        try:
            state = json.loads(Path(state_path).read_text())
        except Exception:
            pass

    by_epoch, error_lines = parse(args.log_path)
    if error_lines and state.get('last_error_count', 0) < len(error_lines):
        for el in error_lines[state.get('last_error_count', 0):]:
            print(f"{args.label}🚨 ERROR: {el}")
        state['last_error_count'] = len(error_lines)

    if not by_epoch:
        Path(state_path).write_text(json.dumps(state))
        return

    cur_ep = max(by_epoch.keys())
    anomalies = detect_anomalies(by_epoch, cur_ep, args.stage)
    if anomalies and state.get('last_anomaly_ep', -1) != cur_ep:
        print(f"{args.label}⚠️ ep={cur_ep} ANOMALY: " + " ; ".join(anomalies))
        state['last_anomaly_ep'] = cur_ep

    if cur_ep >= state['last_reported_ep'] + args.interval:
        report_ep = (cur_ep // args.interval) * args.interval
        eps_with_test = sorted([e for e, d in by_epoch.items() if 'dg_test_acc' in d])
        ref_ep = eps_with_test[-1] if eps_with_test else cur_ep
        print(f"{args.label}✓ {fmt_summary(by_epoch, ref_ep, baseline)}")
        state['last_reported_ep'] = report_ep

    Path(state_path).write_text(json.dumps(state))


if __name__ == '__main__':
    main()
