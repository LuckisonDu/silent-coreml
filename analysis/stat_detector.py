"""ICCE 10-09 内审 v57 ②：「更便宜的检测器」——只看输出的统计量（RMS / 最大绝对值 / 全零比例），能不能分出损坏输出。
固定输入下，正确输出几乎逐位相同，所以「与同输入正确输出的统计量比」是这类检测器的上限（实际出片每次输入不同，没有这个参考值）。
按探针档分组（每个文件里「开始 档=」后面的调用归该档），同一行只算一次。用法: python3 stat_detector.py logs → stat_detector.json"""
import re, glob, sys, json, statistics as st
root = sys.argv[1]; seen = set(); rows = []
for f in sorted(glob.glob(f'{root}/*/abort_probe.txt')):
    tier = None
    for l in open(f, errors='ignore'):
        m = re.search(r'abort-probe 开始 档=(\S+)', l)
        if m: tier = m.group(1); continue
        if 'abort-probe i=' not in l or 'rms=' not in l or 'ref_1mcos=' not in l: continue
        k = l.split('] ', 1)[-1].strip() + '|' + str(tier)
        if k in seen: continue
        seen.add(k)
        g = lambda n: re.search(n + r'=([^ /]+)', l)
        try: rows.append(dict(tier=tier, c=float(g('ref_1mcos').group(1)), rms=float(g('rms').group(1)), amax=float(g('amax').group(1)), zero=float(g('zero').group(1))))
        except Exception: pass
out = {}
for t in sorted({r['tier'] for r in rows}):
    sel = [r for r in rows if r['tier'] == t]; ok = [r for r in sel if r['c'] < 1e-6]; bad = [r for r in sel if r['c'] > 1e-3 and r['zero'] < 0.4]
    if len(ok) < 5 or not bad: continue
    r0 = st.median(r['rms'] for r in ok); a0 = st.median(r['amax'] for r in ok)
    dr = lambda r: abs(r['rms'] / r0 - 1); da = lambda r: abs(r['amax'] / a0 - 1)
    thr_r = max(dr(r) for r in ok); thr_a = max(da(r) for r in ok)
    out[t] = dict(n_ok=len(ok), n_bad_nonzero=len(bad), rms_ref=r0,
                  rms_dev_bad_median=st.median(dr(r) for r in bad), rms_dev_bad_min=min(dr(r) for r in bad),
                  rms_dev_ok_max=thr_r, caught_rms_at_zeroFA=sum(dr(r) > thr_r for r in bad),
                  caught_rms_dev_gt_1pct=sum(dr(r) > 0.01 for r in bad), caught_rms_dev_gt_5pct=sum(dr(r) > 0.05 for r in bad),
                  amax_dev_ok_max=thr_a, caught_amax_at_zeroFA=sum(da(r) > thr_a for r in bad))
    print(t, json.dumps(out[t]))
json.dump(out, open(f'{root}/stat_detector.json', 'w'), indent=1)

# ★10-09 实用阈值(block0_rms_spread.py 在服务器上实测: 正确输出整体 RMS 同步不同片最多偏 2.2% ⇒ 按步标定 x=3%; 全局一个期望值要 x≈7%)
#   注: 这里的「偏离」是相对同输入正确输出, 按步标定时还会叠加最多 ±2.2% 的片间波动, 所以下面的检出数是偏乐观的。
prac = {}
for t in sorted({r['tier'] for r in rows}):
    sel = [r for r in rows if r['tier'] == t]; ok = [r for r in sel if r['c'] < 1e-6]; bad = [r for r in sel if r['c'] > 1e-3 and r['zero'] < 0.4]
    if len(ok) < 5 or len(bad) < 20: continue
    r0 = st.median(r['rms'] for r in ok); d = [abs(r['rms'] / r0 - 1) for r in bad]
    prac[t] = {'n_bad': len(bad), 'caught_x3pct_perstep': sum(x > 0.03 for x in d), 'caught_x7pct_global': sum(x > 0.07 for x in d)}
    print('practical', t, prac[t])
json.dump({'oracle': out, 'practical': prac}, open(f'{root}/stat_detector.json', 'w'), indent=1)
