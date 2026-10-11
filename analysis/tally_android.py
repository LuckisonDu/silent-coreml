"""汇总安卓探针日志(abort_probe_android.txt): 次数、坏几次(哪种坏)、每次耗时、温度、内存。用法: python3 tally_android.py <运行目录>..."""
import re, sys, statistics as S
for d in sys.argv[1:]:
    p = d.rstrip("/") + "/abort_probe_android.txt"
    rows = [l for l in open(p, encoding="utf-8", errors="replace") if "android-probe i=" in l]
    kv = lambda l, k: (re.search(rf"\b{k}=([^\s\]]+)", l) or [None, None])[1]
    dts = [float(kv(l, "dt").rstrip("s")) for l in rows]
    bad = [l for l in rows if kv(l, "bad") == "1"]
    kinds = dict(threw=sum(kv(l, "threw") != "-" for l in rows), sentinel=sum(kv(l, "sentinel") != "0" for l in rows),
                 naninf=sum(kv(l, "naninf") != "0" for l in rows), wrong=sum(kv(l, "bad") == "1" and kv(l, "threw") == "-" and kv(l, "sentinel") == "0" and kv(l, "naninf") == "0" for l in rows))
    aps = [float(x) for l in rows for x in re.findall(r"pw=\[tstat=\d+ AP=([\d.]+)", l)]
    other = [l.strip()[:200] for l in open(p, encoding="utf-8", errors="replace") if "mem-guard" in l or "失败" in l or "结束" in l]
    print(f"== {d}\n  调用 {len(rows)} 次, 坏 {len(bad)} 次 {kinds}")
    if dts: print(f"  每次 {min(dts):.1f}~{max(dts):.1f}s 中位 {S.median(dts):.1f}s; AP 温度 {min(aps, default=0):.1f}~{max(aps, default=0):.1f}°C")
    for l in bad[:5]: print("  坏:", " ".join(f"{k}={kv(l,k)}" for k in ("i", "in", "dt", "threw", "sentinel", "ref_1mcos", "badtok", "first", "last", "tail64", "same_prev")))
    for l in other: print("  ", l)
