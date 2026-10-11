"""末尾校验真机开销：每次调用「校验耗时 tdt / 整块耗时 dt」取中位（ICCE v20，回应第七位评审「2.6/19 = 13.7%」）。
用法: python3 tail_overhead.py logs  → 写 tail_overhead.json"""
import sys, re, json, statistics as st
R = sys.argv[1]; out = {}
for d, ts, tier in [('probephoneAtail_1007_0003', '07:13', 'fp32_A'), ('probephoneAtail_1007_0003', '08:12', 'shipped_A'), ('probephoneBtailgate_1007_0506', '12:14', 'fp32_B'),
                    ('probephoneCtailgate_1007_0500', '12:09', 'fp32_C'), ('probephoneAshipped_1007_0753', '15:00', 'shipped_A2'), ('probephoneBshipped_1007_0800', '15:08', 'shipped_B2'), ('probephoneAb16_1007_0337', '10:48', 'b16_A')]:
    L = open(f'{R}/{d}/abort_probe.txt').read().splitlines()
    k = [n for n, l in enumerate(L) if '开始' in l and l.startswith('[2026-10-07 ' + ts)][0]; rows = []
    for l in L[k + 1:]:
        if '开始' in l: break
        m = re.search(r' dt=([\d.]+)s.*tdt=([\d.]+) tload=([\d.]+)', l)
        if m: rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    out[tier] = dict(n=len(rows), dt_med=round(st.median(r[0] for r in rows), 2), tdt_med=round(st.median(r[1] for r in rows), 2), ratio_med=round(st.median(r[1] / r[0] for r in rows), 3), ratio_with_load_med=round(st.median((r[1] + r[2]) / r[0] for r in rows), 3))
json.dump(out, open(f'{R}/tail_overhead.json', 'w'), indent=1); print(json.dumps(out, indent=1))
