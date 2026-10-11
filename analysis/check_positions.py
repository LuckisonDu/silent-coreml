"""ICCE 10-09 外审追问「同样开销下，只查末尾 vs 随机抽查 vs 均匀分散抽查」：用全部已记录的非零静默错误输出（与 corrupt_structure.py 同一筛法）
模拟四种只重算 64 个 token 的查法能否检出。每个输出只有 bad/first/last 三个数，故：
  tail64   : 检出 ⇔ last ≥ T-64（末尾 64 个里至少一个坏 token；首坏到末坏之间按日志几乎全坏，见 suffix_bad_frac）
  head64   : 检出 ⇔ first < 64
  stride64 : 64 个位置 p_k = floor((k+0.5)·T/64)；检出 ⇔ 至少一个 p_k 落在 [first, last] 内（坏区间内的密度另报，作为保守性说明）
  random64 : 检出概率 = 1 − C(T−bad, 64)/C(T, 64)（期望漏检 = Σ 漏检概率）
按芯片分组：目录名含 iphone18pro 的是 iPhone 18（A20，诱发 GPU 恢复），其余 iPhone 12（A14，自然发热）；分段导出单列。
用法: python3 check_positions.py logs → check_positions.json"""
import re, glob, sys, json, math
root = sys.argv[1]; seen = set(); rows = []
for f in sorted(glob.glob(f'{root}/*/abort_probe.txt')):
    src = f.split('/')[-2]
    for l in open(f, errors='ignore'):
        l = l.strip()
        if 'abort-probe i=' not in l or 'badtok=' not in l or l in seen: continue
        seen.add(l)
        m = re.search(r'ref_1mcos=([^ ]+).*?badtok=(\d+)/(\d+) first=(-?\d+) last=(-?\d+)', l)
        if not m or float(m.group(1)) < 1e-3: continue
        z = re.search(r'zero=([\d.]+)', l)
        if z and float(z.group(1)) > 0.4: continue
        bad, T, first, last = map(int, m.group(2, 3, 4, 5))
        if bad == 0 or first < 0: continue
        rows.append(dict(src=src, T=T, bad=bad, first=first, last=last, split='seg=' in l, chip='A20' if 'iphone18pro' in src else 'A14'))


def lgC(n, k): return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


K = 64; out = {}
for name, sel in [('A14_whole', [r for r in rows if r['chip'] == 'A14' and not r['split']]), ('A14_split', [r for r in rows if r['chip'] == 'A14' and r['split']]),
                  ('A20_whole', [r for r in rows if r['chip'] == 'A20' and not r['split']])]:
    if not sel: continue
    tail = sum(r['last'] >= r['T'] - K for r in sel)
    head = sum(r['first'] < K for r in sel)
    stride = 0; dens = []; stride_miss = 0.0
    for r in sel:
        T = r['T']; ps = [int((k + 0.5) * T / K) for k in range(K)]
        inside = sum(r['first'] <= p <= r['last'] for p in ps); stride += inside > 0
        d = r['bad'] / (r['last'] - r['first'] + 1); dens.append(d)
        stride_miss += (1 - min(d, 1.0)) ** inside if inside else 1.0   # 坏 token 在区间内随机分布时, 落入区间的抽查点全落在好 token 上的概率
    miss_rand = sum(math.exp(lgC(r['T'] - r['bad'], K) - lgC(r['T'], K)) if r['T'] - r['bad'] >= K else 0.0 for r in sel)
    span_min = min(r['last'] - r['first'] + 1 for r in sel)
    out[name] = dict(n=len(sel), tail64=tail, head64=head, stride64=stride, random64_expected_misses=round(miss_rand, 3), stride64_expected_misses=round(stride_miss, 3), low_density_n=sum(x < 0.9 for x in dens),
                     min_corrupt_span_tokens=span_min, span_density_min=round(min(dens), 3), span_density_median=round(sorted(dens)[len(dens) // 2], 3),
                     stride_spacing_tokens=round(sel[0]['T'] / K, 1))
    print(name, out[name])
json.dump(out, open(f'{root}/check_positions.json', 'w'), ensure_ascii=False, indent=1)
