"""预防实验 C 统计：按「开始 档=…」切段，标出整块 / 三段+段间空隙，统计错次数、被掐段、耗时。用法: python3 gap_tally.py <abort_probe.txt> [起始UTC时间串]"""
import re, sys, collections, statistics as st
L = open(sys.argv[1]).read().splitlines(); since = sys.argv[2] if len(sys.argv) > 2 else '[2026-10-06 19:5'
starts = [i for i, l in enumerate(L) if 'abort-probe 开始 档=' in l and l > since]
tot = collections.defaultdict(lambda: [0, 0])
for j, si in enumerate(starts):
    end = starts[j + 1] if j + 1 < len(starts) else len(L)
    seg = L[si + 1:end]; rows = [l for l in seg if 'abort-probe i=' in l]
    gap = next((re.search(r'额外停 ([\d.]+)', l).group(1) for l in L[max(0, si - 5):si + 10] if '额外停' in l), '0')
    t = re.search(r'档=(\S+)', L[si]).group(1); key = t if 's3' not in t else f'三段 gap={gap}'
    w = sum(float(re.search(r'ref_1mcos=([^ ]+)', l).group(1)) > 1e-3 for l in rows)
    dts = [float(re.search(r'dt=([\d.]+)', l).group(1)) for l in rows]
    ps = [re.search(r'seg=(\d)', l).group(1) for l in seg if 'postseg' in l]
    print(L[si][12:20], key, len(rows), '错', w, '被掐段', ps, 'dt 中位 %.1f' % st.median(dts) if dts else '')
    if j > 0: tot[key][0] += w; tot[key][1] += len(rows)
print({k: f'{v[0]}/{v[1]}' for k, v in tot.items()})
