"""ICCE 10-15：把全部不同的已记录探测输出（abort_probe.txt 里的 abort-probe i= 行，去重）分成互不重叠、合起来等于总数的几组，供论文「总结果表」用。
分组键：设备 / 模型与导出档 / 条件（自然或界面刷新 / 诱发 GPU 恢复）/ 设计前后（10-09 10:20 之前的目录算设计前）。损坏 = 1−cos>THR。
设备由目录名判定（xichu、x4、x5split=手机 A；xiaodong、xd*=手机 B；lucki、lk*=手机 C；x16=iPhone 16；tan18=iPhone 18 Pro）。
用法: 改开头「在这里改」里的变量后直接运行（C1180：不用命令行参数） → ROOT/outputs_partition.json"""
import re, glob, json, collections
# ===== 在这里改 =====
ROOT = '/path/to/your/copy/of/the/logs'   # 读其下 probe*/abort_probe.txt，结果写 ROOT/outputs_partition.json
THR = 1e-3                                  # 判损坏阈值（1−cos）
CUT = ('1009', '1020')                      # 设计冻结（目录名里的 MMDD_HHMM）：此前算设计前
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


seen = set(); G = collections.defaultdict(lambda: [0, 0]); other = collections.Counter()
for f in sorted(glob.glob(f'{ROOT}/probe*/abort_probe.txt'), key=lambda f: (key(f.split('/')[-2]), f)):
    d = f.split('/')[-2]
    lines = open(f, errors='ignore').read().split('\n')
    # 按「abort-probe 开始」切成运行段；段内有「卡核」行 = 诱发 GPU 恢复；第一段之前的行（早期无标记的探针）算 fp32 块 0 的自然运行
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
            tc = tierclass(sg['tier'])
            if tc.startswith('other'): other[tc] += 1
            cond = 'induced GPU recovery' if hang else ('UI refreshing' if sg['ui'] == 'true' else 'natural')
            phase = 'post' if key(d) >= CUT else 'pre'
            g = (dev(d), tc, cond, phase); G[g][0] += 1; G[g][1] += v > THR
rows = [dict(device=g[0], model=g[1], condition=g[2], phase=g[3], n=n, corrupted=c) for g, (n, c) in sorted(G.items())]
out = dict(total=sum(r['n'] for r in rows), corrupted=sum(r['corrupted'] for r in rows), rows=rows, other=other)
json.dump(out, open(f'{ROOT}/outputs_partition.json', 'w'), ensure_ascii=False, indent=1)
print(out['total'], out['corrupted'], dict(other))
for r in rows: print(r)
