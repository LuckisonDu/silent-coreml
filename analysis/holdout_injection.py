"""ICCE 10-09 内审 v59 ①②（离线，不用手机 / GPU）。
① 发现集 / 留出集：分散抽查的位置公式 p_k=floor((k+0.5)·T/K) 与 K=64 是 10-09 定的；把已记录的非零损坏输出按「时间」（10-08 00:00 PDT 之前 / 之后）
   和按「手机」（phoneA = 发现，其余 = 留出）各分两半，只在留出那半报检出（与 check_positions.py 同一筛法、同一判据：位置落在 [first,last] 内即检出，
   区段内坏 token 稀疏的按密度给期望值）。
② 故障注入（纯组合计算，不碰张量）：T=11440，K∈{16,32,64,128}。
   a) 连续全错区段，长 L，起点均匀随机：分散检出概率 = 覆盖到至少一个抽查位置的起点比例（精确枚举）；末尾检查只在区段碰到最后 K 个 token 时检出。
      覆盖界：任何长度 ≥ ceil(T/K) 的连续全错区段必被分散抽查命中。
   b) 零散错 token：n 个随机位置出错，P(命中) = 1 − C(T−K, n)/C(T, n)。
   c) 稀疏区段：长 L、区段内每个 token 以概率 d 出错：P = 1 − (1−d)^(区段内抽查位置数)，对起点平均。
用法: python3 holdout_injection.py logs → holdout_injection.json"""
import re, glob, sys, json, math
# --paper-v60: 复现论文 v60 的「按手机」留出数(514)。v60 当时按目录名认机, 早期两类 Phone A 目录(原名 probex4 / probex5split, 本包里是
#   probephoneA4_* / probephoneA5split_*)名字里没有机名, 被归到「其他」= 留出集; 不加此开关则按实际机器归 Phone A(发现集), 留出 489。
V60 = '--paper-v60' in sys.argv; sys.argv = [a for a in sys.argv if a != '--paper-v60']
root = sys.argv[1]; seen = set(); rows = []
for f in sorted(glob.glob(f'{root}/*/abort_probe.txt')):
    src = f.split('/')[-2]; ph = next((p for p in ('phoneA', 'phoneB', 'phoneC', 'iphone18pro') if p in src), 'other')
    if V60 and src.startswith(('probephoneA4_', 'probephoneA5')): ph = 'other'
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
        ts = re.match(r'\[(\d{4}-\d\d-\d\d) (\d\d)', l); utc = ts.group(1) + 'T' + ts.group(2) if ts else ''
        rows.append(dict(T=T, bad=bad, first=first, last=last, phone=ph, utc=utc, split='seg=' in l, chip='A20' if ph == 'iphone18pro' else 'A14'))

def p_stride(r, K):
    ps = [int((k + 0.5) * r['T'] / K) for k in range(K)]; ins = sum(r['first'] <= p <= r['last'] for p in ps)
    d = min(1.0, r['bad'] / (r['last'] - r['first'] + 1)); return 1 - (1 - d) ** ins if ins else 0.0
def summ(sel, K=64):
    return dict(n=len(sel), tail=sum(r['last'] >= r['T'] - K for r in sel), stride_expected=round(sum(p_stride(r, K) for r in sel), 2),
                stride_any_position_in_span=sum(any(r['first'] <= int((k + .5) * r['T'] / K) <= r['last'] for k in range(K)) for r in sel))
out = {'holdout': {}}
cut = '2026-10-08T07'      # 10-08 00:00 PDT = 07:00 UTC
for name, a, b in [('by_time', [r for r in rows if r['utc'] < cut], [r for r in rows if r['utc'] >= cut]),
                   ('by_phone', [r for r in rows if r['phone'] == 'phoneA'], [r for r in rows if r['phone'] != 'phoneA'])]:
    out['holdout'][name] = dict(discovery=summ(a), holdout=summ(b), holdout_by_group={g: summ([r for r in b if (r['chip'], r['split']) == k]) for g, k in
                                [('A14_whole', ('A14', False)), ('A14_split', ('A14', True)), ('A20', ('A20', False))]})
    print(name, json.dumps(out['holdout'][name]))
T = 11440; inj = {}
for K in (16, 32, 64, 128):
    ps = sorted({int((k + .5) * T / K) for k in range(K)}); ent = {'bound_any_contiguous_span_len_ge': math.ceil(T / K)}
    for L in (16, 32, 64, 100, 179, 256, 512, 918, 1430):
        starts = range(0, T - L + 1); n = len(starts); hit = tail = 0; j = 0
        for s in starts:
            hit += any(s <= p < s + L for p in ps[max(0, j - 2):]) if False else (next((p for p in ps if p >= s), T) < s + L)
            tail += (s + L > T - K)
        ent[f'contig_L{L}'] = dict(stride=round(hit / n, 4), tail=round(tail / n, 4))
    for nn in (1, 5, 20, 100):
        ent[f'scatter_n{nn}'] = round(1 - math.exp(math.lgamma(T - K + 1) - math.lgamma(T - K - nn + 1) - math.lgamma(T + 1) + math.lgamma(T - nn + 1)), 4)
    for L, d in ((1430, 0.2), (918, 0.66), (512, 0.5), (256, 0.5)):
        tot = 0.0; n = T - L + 1
        for s in range(n): ins = sum(s <= p < s + L for p in ps); tot += 1 - (1 - d) ** ins
        ent[f'sparse_L{L}_d{d}'] = round(tot / n, 4)
    inj[K] = ent; print('K', K, json.dumps(ent))
out['injection_T11440'] = inj
json.dump(out, open(f'{root}/holdout_injection.json', 'w'), indent=1)
