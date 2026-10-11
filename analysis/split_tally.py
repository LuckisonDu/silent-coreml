"""预防实验统计：按「开始 档=…」切段，统计每段 run 的对/错/日志记录/耗时。用法: python3 split_tally.py <abort_probe.txt>"""
import re, sys, collections, statistics as st
L = open(sys.argv[1]).read().splitlines()
starts = [i for i, l in enumerate(L) if 'abort-probe 开始 档=' in l]
tot = collections.defaultdict(collections.Counter)
for j, si in enumerate(starts):
    if '10-06' not in L[si] and '2026-10-06' not in L[si]: continue
    tier = re.search(r'档=(\S+)', L[si]).group(1)
    if L[si] < '[2026-10-06 18:07': continue           # 只算 v2 修好之后（18:07 UTC 起）这一轮
    seg = L[si + 1: starts[j + 1] if j + 1 < len(starts) else len(L)]
    rows = [l for l in seg if 'abort-probe i=' in l]
    c = collections.Counter(); dts = []
    for l in rows:
        w = float(re.search(r'ref_1mcos=([^ ]+)', l).group(1)) > 1e-3
        zf = float(re.search(r'zero=([\d.]+)', l).group(1))
        m = re.search(r'影响交互 (\d+) / 连带丢弃 (\d+)', l); k = 'none' if not m else ('II' if int(m.group(1)) > 0 else 'IV')
        form = 'ok' if not w else ('allzero' if zf > 0.99 else 'nonzero')
        c[(k, form)] += 1; dts.append(float(re.search(r'dt=([\d.]+)', l).group(1)))
        tot[tier][(k, form)] += 1
    if rows: print(L[si][1:20], tier, 'n', len(rows), dict(c), 'dt 中位 %.1f 最大 %.1f' % (st.median(dts), max(dts)))
print('合计:'); [print(' ', t, sum(v.values()), dict(v)) for t, v in tot.items()]
