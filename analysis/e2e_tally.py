"""端到端三组统计：trace.txt 是历次累加的，取最后一次「步数 4 σ=」之后那段（=本条片）。
数: 掐断(检测到但未处理, none/zero 组) / 日志判废回退(log 组, '作废(GPU') / 全零判废(作废(精确 0) / 出片总时长。
用法: python3 e2e_tally.py logs/e2e"""
import sys, os, re, glob, json
root = sys.argv[1]; out = []
for d in sorted(glob.glob(f'{root}/*_s*_*')):
    t = os.path.join(d, 'trace.txt')
    if not os.path.exists(t): continue
    L = open(t, errors='ignore').read().splitlines()
    idx = [i for i, l in enumerate(L) if '步数 ' in l and 'σ=' in l] or [i for i, l in enumerate(L) if l.startswith('[   0.0s')]
    if not idx: print('跳过(无起点)', d); continue
    st = idx[-1]
    seg = L[st:]
    unh = sum('⚠️(未处理' in l for l in seg)
    rb_log = sum('作废(GPU' in l for l in seg)
    rb_zero = sum('作废(精确 0' in l or '作废(非有限' in l for l in seg)
    last = [l for l in seg if re.match(r'\[\s*[\d.]+s', l)]
    tot = float(re.match(r'\[\s*([\d.]+)s', last[-1]).group(1)) if last else None
    dev, s, arm = os.path.basename(d).rsplit('_', 2)[0], os.path.basename(d).split('_')[1], os.path.basename(d).split('_')[-1]
    out.append(dict(clip=os.path.basename(d), arm=arm, unhandled_aborts=unh, log_reruns=rb_log, zero_reruns=rb_zero,
                    trace_seconds=tot, mp4=os.path.exists(os.path.join(d, 'auto_output.mp4')) or os.path.exists(os.path.join(d, 'auto_output.framemd5'))))
for r in out: print(r)
json.dump(out, open(f'{root}/e2e_tally.json', 'w'), ensure_ascii=False, indent=1)
