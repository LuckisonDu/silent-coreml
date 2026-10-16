"""ICCE 10-15：逐次运行的明细（公开仓库 runs_summary.csv）：每个运行目录 × 模型/导出 × 条件 一行，含开始时间（目录名里的 MMDD_HHMM，太平洋时间）、设备代号、输出数、损坏数。
与 outputs_partition.py 同一套去重/分类（累计设备日志里同一行只算在最先出现的目录），所以逐行相加等于论文表 I 的 10,315（1,250）；10 月 15 日各轮（postdesign_1015/*/summary.json）另列在后。
只输出计数，不含原始日志行、设备昵称、路径。用法: 改开头「在这里改」里的变量后直接运行（C1180：不用命令行参数） → OUT。"""
import re, glob, csv, json, collections
# ===== 在这里改 =====
ROOT = '/path/to/your/copy/of/the/logs'
OUT = 'runs_summary.csv'
THR = 1e-3
CUT = ('1009', '1020')
# ====================
DEV = [('tan18', 'iPhone 18 Pro'), ('x16', 'iPhone 16'), ('xiaodong', 'Phone B'), ('xd', 'Phone B'), ('lucki', 'Phone C'), ('lk', 'Phone C'),
       ('xichu', 'Phone A'), ('x4', 'Phone A'), ('x5split', 'Phone A')]


def dev(d):
    for k, v in DEV:
        if k in d: return v
    return '?'


def key(d):
    m = re.search(r'_(\d{4})_(\d{4})', d)
    return (m.group(1), m.group(2)) if m else ('0000', '0000')


def tierclass(t):
    if t.startswith('klein'): return 'Klein block'
    if t.startswith('ltx'): return 'LTX-2.3 block 0'
    if t.startswith('576'): return 'Wan 576 px'
    if t.startswith('704x1280m'): return 'Wan shipped export'
    if 'b16' in t: return 'Wan fp32 block 16'
    if re.search(r'f32s3', t): return 'Wan fp32 split calls'
    if re.search(r'f32(c\d|q\d)', t): return 'Wan fp32 chain/query-chunk variants'
    if t.startswith('704x1280f32p'): return 'Wan fp32 block 0'
    return 'other:' + t


seen = set(); G = collections.OrderedDict()
for f in sorted(glob.glob(f'{ROOT}/probe*/abort_probe.txt'), key=lambda f: (key(f.split('/')[-2]), f)):
    d = f.split('/')[-2]; k0 = key(d)
    lines = open(f, errors='ignore').read().split('\n')
    segs = []; cur = dict(tier='704x1280f32p', ui='false', lines=[]); segs.append(cur)
    for l in lines:
        m = re.search(r'abort-probe 开始 档=(\S+) iters=(\d+).*?ui压力=(\w+)', l)
        if m: cur = dict(tier=m.group(1), ui=m.group(3), lines=[]); segs.append(cur); continue
        cur['lines'].append(l)
    for sg in segs:
        hang = any('卡核' in l for l in sg['lines'])
        for l in sg['lines']:
            if 'abort-probe i=' not in l or l in seen: continue
            seen.add(l); mm = re.search(r'ref_1mcos=([^ ]+)', l)
            try: v = float(mm.group(1))
            except Exception: continue
            cond = 'induced GPU recovery' if hang else ('UI refreshing' if sg['ui'] == 'true' else 'natural')
            g = (k0, dev(d), tierclass(sg['tier']), cond, 'after design' if k0 >= CUT else 'before design')
            G.setdefault(g, [0, 0]); G[g][0] += 1; G[g][1] += v > THR
rows = [dict(start_pdt=f'2026-{g[0][0][:2]}-{g[0][0][2:]} {g[0][1][:2]}:{g[0][1][2:]}', device=g[1], model_export=g[2], condition=g[3], phase=g[4], outputs=n, corrupted=c)
        for g, (n, c) in G.items()]
rows.sort(key=lambda r: (r['start_pdt'], r['device'], r['model_export'], r['condition']))
tot = (sum(r['outputs'] for r in rows), sum(r['corrupted'] for r in rows))
# 10 月 15 日各轮（不在 10,315 里）
extra = []
for name, dv, cond, note in [('xichu', 'Phone A', 'natural (hot start)', 'first Oct. 15 run'), ('xichu_r2', 'Phone A', 'natural (hot start)', 'second Oct. 15 run'),
                             ('x16_r2', 'iPhone 16', 'induced GPU recovery', 'second Oct. 15 run')]:
    p = f'{ROOT}/postdesign_1015/{name}/summary.json'
    try: s = json.load(open(p))
    except Exception: continue
    extra.append(dict(start_pdt='2026-10-15', device=dv, model_export='Wan fp32 block 0', condition=cond, phase='after design (not in Table I)', outputs=s.get('calls'), corrupted=s.get('corrupted')))
extra.insert(2, dict(start_pdt='2026-10-15', device='iPhone 16', model_export='Wan fp32 block 0', condition='induced GPU recovery', phase='after design (not in Table I)', outputs=51, corrupted=11))   # 10-15 第一轮（Table II：iPhone 16 共 103(20) = 51(11) + 52(9)）
with open(OUT, 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows); w.writerows(extra)
print(len(rows), tot, 'extra', extra)
