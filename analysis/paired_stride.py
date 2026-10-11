"""ICCE 10-09: 同一调用里末尾校验(tailchk)与分散校验(tail2chk)成对实测的计数。判损坏: badtok>0（有逐 token 错；无该字段时 ref_1mcos>1e-5）——与 10-09 手算口径一致（iphone18pro i=46 只 1.6e-4 但 1974 个坏 token，算损坏）; 判标出: 1−cos>1e-4。
只算带 tail2chk 的行(--abort-probe-tail2, 10-09 新加), 同一行去重。用法: python3 paired_stride.py <abort_probe.txt>..."""
import re, sys, json
seen = set(); n = bad = t1 = t2 = f1 = f2 = 0
for f in sys.argv[1:]:
    for l in open(f, errors='ignore'):
        if 'abort-probe i=' not in l or 'tail2chk=' not in l or l in seen: continue
        seen.add(l); g = lambda k: float(re.search(k + r'=([^ /]+)', l).group(1))
        try: c, a, b = g('ref_1mcos'), g('tailchk') > 1e-4, g('tail2chk') > 1e-4
        except Exception: continue
        n += 1
        m = re.search(r'badtok=(\d+)/', l); corrupt = int(m.group(1)) > 0 if m else c > 1e-5
        if corrupt: bad += 1; t1 += a; t2 += b
        else: f1 += a; f2 += b
r = dict(calls=n, corrupted=bad, tail_flagged=t1, strided_flagged=t2, tail_fa=f1, strided_fa=f2, correct=n - bad); print(json.dumps(r))
