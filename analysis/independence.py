"""ICCE 10-09（外部意见「1047 个损坏样本的独立性」）：分散检查离线检出的 1047 个非零损坏输出（与 check_positions.py 同一筛法）
来自多少次独立运行 / 几台手机 / 几种条件 / 几天；错误是不是同一个模式反复出现（区段起点、区段长度有多少种）；按「运行」为单位的漏检上限。
运行 = 一个探针目录里一次「开始 档=」到下一次之间（一次 App 启动内的连续调用）。用法: python3 independence.py logs → independence.json"""
import re, glob, sys, json, collections, math
root = sys.argv[1]; seen = set(); rows = []
for f in sorted(glob.glob(f'{root}/*/abort_probe.txt')):
    src = f.split('/')[-2]; ph = next((p for p in ('phoneA', 'phoneB', 'phoneC', 'iphone18pro') if p in src), 'other'); run = None; k = 0
    for l in open(f, errors='ignore'):
        l = l.strip()
        m0 = re.search(r'(\[\d{4}-\d\d-\d\d [0-9:]+).*abort-probe 开始 档=(\S+)', l)
        if m0: run = f'{src}|{m0.group(1)}|{m0.group(2)}'; continue
        if 'abort-probe i=' not in l or 'badtok=' not in l or l in seen: continue
        seen.add(l)
        m = re.search(r'ref_1mcos=([^ ]+).*?badtok=(\d+)/(\d+) first=(-?\d+) last=(-?\d+)', l)
        if not m or float(m.group(1)) < 1e-3: continue
        z = re.search(r'zero=([\d.]+)', l)
        if z and float(z.group(1)) > 0.4: continue
        bad, T, first, last = map(int, m.group(2, 3, 4, 5))
        if bad == 0 or first < 0: continue
        d = re.match(r'\[(\d{4}-\d\d-\d\d)', l)
        grp = 'A20' if ph == 'iphone18pro' else ('A14_split' if 'seg=' in l else 'A14_whole')
        rows.append(dict(run=run or src, phone=ph, grp=grp, day=d.group(1) if d else '', T=T, first=first, last=last, span=last - first + 1))
out = dict(n=len(rows))
runs = collections.Counter(r['run'] for r in rows)
out['independent_runs'] = len(runs); out['phones'] = dict(collections.Counter(r['phone'] for r in rows))
out['days_utc'] = dict(sorted(collections.Counter(r['day'] for r in rows).items()))
out['runs_per_group'] = {g: len({r['run'] for r in rows if r['grp'] == g}) for g in ('A14_whole', 'A14_split', 'A20')}
out['n_per_group'] = dict(collections.Counter(r['grp'] for r in rows))
cs = sorted(runs.values(), reverse=True); out['per_run_count'] = dict(max=cs[0], median=cs[len(cs) // 2], top5=cs[:5], share_top5=round(sum(cs[:5]) / len(rows), 3))
out['distinct_first_positions'] = len({(r['T'], r['first']) for r in rows}); out['distinct_spans'] = len({(r['T'], r['first'], r['last']) for r in rows})
top = collections.Counter((r['T'], r['first'], r['last']) for r in rows).most_common(5); out['most_common_spans'] = [[list(k), v] for k, v in top]
out['span_len_quantiles'] = {q: sorted(r['span'] for r in rows)[int(q * (len(rows) - 1))] for q in (0.0, 0.1, 0.5, 0.9, 1.0)}
R = len(runs); out['run_level_zero_miss_upper95'] = round(1 - 0.05 ** (1 / R), 4)   # 0/R 次运行有漏检 ⇒ 每次运行出现漏检的概率 95% 上限
out['note'] = '离线：分散位置是否落进已记录区段；不是手机上真跑校验器的检出'
print(json.dumps(out, ensure_ascii=False, indent=1)); json.dump(out, open(f'{root}/independence.json', 'w'), ensure_ascii=False, indent=1)
