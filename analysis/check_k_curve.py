"""ICCE 10-09 内审③: 检出随重算 token 数 K 的变化（K=8…256），四种放法：末尾 K / 均匀分散 K / 末尾 K/2 + 分散 K/2 / 随机 K（期望）。
数据与 check_positions.py 同一筛法（全部已记录非零静默错误输出，按 A14 整块 / A14 拆分 / A20 诱发分组）。
末尾：检出 ⇔ last ≥ T−K；分散：检出概率 = 1 − (1−d)^(落入 [first,last] 的抽查点数)，d = 区段内坏 token 比例；随机：1 − C(T−bad,K)/C(T,K)。
输出期望检出数（分散/随机为期望值）。用法: python3 check_k_curve.py logs → check_k_curve.json"""
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
        rows.append(dict(T=T, bad=bad, first=first, last=last, split='seg=' in l, chip='A20' if 'iphone18pro' in src else 'A14'))


def lgC(n, k): return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def p_stride(r, K):
    if K <= 0: return 0.0
    ps = [int((k + 0.5) * r['T'] / K) for k in range(K)]
    inside = sum(r['first'] <= p <= r['last'] for p in ps)
    d = min(1.0, r['bad'] / (r['last'] - r['first'] + 1))
    return 1 - (1 - d) ** inside if inside else 0.0


def p_rand(r, K): return 1 - (math.exp(lgC(r['T'] - r['bad'], K) - lgC(r['T'], K)) if r['T'] - r['bad'] >= K else 0.0)


out = {}
groups = [('A14_whole', [r for r in rows if r['chip'] == 'A14' and not r['split']]), ('A14_split', [r for r in rows if r['chip'] == 'A14' and r['split']]),
          ('A20_induced', [r for r in rows if r['chip'] == 'A20' and not r['split']])]
for name, sel in groups:
    out[name] = {'n': len(sel)}
    for K in [8, 16, 32, 64, 128, 256]:
        tail = sum(r['last'] >= r['T'] - K for r in sel)
        stride = sum(p_stride(r, K) for r in sel)
        mix = sum(1 - (1 - (1.0 if r['last'] >= r['T'] - K // 2 else 0.0)) * (1 - p_stride(r, K // 2)) for r in sel)
        rand = sum(p_rand(r, K) for r in sel)
        out[name][K] = dict(tail=tail, stride=round(stride, 2), tail_plus_stride=round(mix, 2), random=round(rand, 2))
    print(name, json.dumps(out[name]))
json.dump(out, open(f'{root}/check_k_curve.json', 'w'), indent=1)
