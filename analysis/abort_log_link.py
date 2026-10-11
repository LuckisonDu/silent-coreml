"""ICCE v27：GPU 中止记录与静默错误一一对应的直接证据（回应外部评审「看门狗只是相关推断」）。
逐次调用：corrupted（ref_1mcos>1e-3）× 有没有 GPU 中止记录（调用窗口内 os_log 的 kIOGPUCommandBufferCallbackError*，
来源 = 调用后即时读到的 oslog= 字段，或事后按时间回填的「post i=」行）。只取单次调用（不含分段 seg=）的整块与上线档探针轮。
用法: python3 abort_log_link.py logs → abort_log_link.json"""
import sys, re, glob, json, collections
R = sys.argv[1]; seen = set(); T = collections.Counter(); per = collections.defaultdict(collections.Counter); dt = collections.defaultdict(list)
for f in sorted(glob.glob(f'{R}/*/abort_probe.txt')):
    tier = None; rows = {}; posts = set(); key = None
    def flush():
        for i, (w, rec, d) in rows.items():
            rec = rec or (i in posts)
            T[(w, rec)] += 1; per[tier][(w, rec)] += 1; dt[(w, rec)].append(d)
    for l in open(f, errors='ignore'):
        if '开始 档=' in l:
            flush(); rows = {}; posts = set(); tier = re.search(r'档=(\S+)', l).group(1); key = l.strip(); continue
        if tier is None or not (tier.startswith('704x1280f32') or tier.startswith('704x1280m')): continue
        if tier.endswith(('s3', 's3v2', 'split2', 'split3', 'split3v2')): continue
        m = re.search(r'abort-probe post i=(\d+) ii=(\d+) iv=(\d+)', l)
        if m: posts.add(int(m.group(1))); continue
        if 'abort-probe i=' not in l or 'ref_1mcos=' not in l or 'seg=' in l: continue
        s = l.strip()
        if s in seen: continue
        seen.add(s)
        i = int(re.search(r'abort-probe i=(\d+)', s).group(1)); c = float(re.search(r'ref_1mcos=([^ ]+)', s).group(1))
        o = re.search(r'oslog=(\S+)', s).group(1); d = float(re.search(r' dt=([\d.]+)s', s).group(1))
        rows[i] = (c > 1e-3, o != '-', d)
    flush()
import statistics as st
out = dict(corrupted_with_record=T[(True, True)], corrupted_without_record=T[(True, False)], correct_with_record=T[(False, True)], correct_without_record=T[(False, False)],
           dt_median={f"{'corrupt' if w else 'ok'}_{'rec' if r else 'norec'}": round(st.median(v), 2) for (w, r), v in dt.items() if v},
           per_tier={t: {f"{'corrupt' if w else 'ok'}_{'rec' if r else 'norec'}": n for (w, r), n in c.items()} for t, c in per.items()})
json.dump(out, open(f'{R}/abort_log_link.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(out, indent=1, ensure_ascii=False))
