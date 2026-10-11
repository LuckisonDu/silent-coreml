"""所有非零静默错误输出的「坏在哪」：按整行去重（手机日志累加、被多个目录重复拉回），统计坏 token 是否一直延续到序列末尾、首坏位置、段数。
用法: python3 corrupt_structure.py logs"""
import re, glob, collections, statistics as st, sys, json
root = sys.argv[1]; seen = set(); rows = []
for f in sorted(glob.glob(f'{root}/*/abort_probe.txt')):
    for l in open(f, errors='ignore'):
        l = l.strip()
        if 'abort-probe i=' not in l or 'badtok=' not in l or l in seen: continue
        seen.add(l)
        m = re.search(r'ref_1mcos=([^ ]+).*?badtok=(\d+)/(\d+) first=(-?\d+) last=(-?\d+) runs=(\d+)', l)
        if not m or float(m.group(1)) < 1e-3: continue
        if float(re.search(r'zero=([\d.]+)', l).group(1)) > 0.4: continue      # 全零与诊断用占位输入不算
        if 'seg=' in l and re.search(r'seg=[\d.]+\s', l): pass
        bad, T, first, last, runs = map(int, m.group(2, 3, 4, 5, 6))
        rows.append(dict(src=f.split('/')[-2], T=T, bad=bad, first=first, last=last, runs=runs, split='seg=' in l))
out = {}
for name, sel in [('整块', [r for r in rows if not r['split']]), ('分段', [r for r in rows if r['split']])]:
    if not sel: continue
    tail = sum(r['last'] == r['T'] - 1 for r in sel)
    sf = sorted((r['bad'] / (r['T'] - r['first']) if r['T'] > r['first'] else 1.0) for r in sel)
    out[name] = dict(n=len(sel), tail_to_end=tail, suffix_bad_frac_median=round(st.median(sf), 3), suffix_bad_frac_p10=round(sf[len(sf) // 10], 3),
                     n_first_after_5pct=sum(r['first'] / r['T'] > 0.05 for r in sel), bad_frac_median=round(st.median(r['bad'] / r['T'] for r in sel), 3),
                     first_pos_median=round(st.median(r['first'] / r['T'] for r in sel), 3),
                     first_pos_min=round(min(r['first'] / r['T'] for r in sel), 3), first_pos_max=round(max(r['first'] / r['T'] for r in sel), 3))
    print(name, out[name])
json.dump(out, open(f'{root}/corrupt_structure.json', 'w'), ensure_ascii=False, indent=1)
