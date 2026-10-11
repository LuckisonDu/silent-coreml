"""跨层证据（论文 Sec. 3 第 (4) 点与 Sec. 6「newer phone」段）：iPhone 18 Pro 上用 --abort-probe-gpuhang 让后台线程反复提交超长 Metal 核、
逼 iOS 做 GPU 恢复。按「开始 档=」切成运行，每个运行报：
  Metal 侧：本运行里我们自己提交的卡核命令缓冲（「卡核 #n」行）个数，及其错误码分布（kIOGPUCommandBufferCallbackError*）；
  Core ML 侧：调用数、损坏数（badtok>0，或无该字段时 ref_1mcos>1e-3）、抛错数（threw≠-）、损坏而无中止日志的个数；
  有末尾/分散校验时：校验标出数（1−cos>1e-4）、误报数。
用法: python3 gpuhang_crosslayer.py logs/probeiphone18proosgpuhang_1009_0102/abort_probe.txt logs/probeiphone18proosgpuhangtail3_1009_0704/abort_probe.txt"""
import re, sys, json, collections
out = {}
for f in sys.argv[1:]:
    L = open(f, errors='ignore').read().splitlines()
    st = [i for i, l in enumerate(L) if 'abort-probe 开始 档=' in l]
    for j, s in enumerate(st):
        seg = L[s + 1:st[j + 1] if j + 1 < len(st) else len(L)]
        hang = [l for l in seg if '卡核 #' in l]
        if not hang: continue
        codes = collections.Counter((re.search(r'(kIOGPUCommandBufferCallbackError\w+)', l) or [None, 'no-error'])[1] for l in hang)
        rows = [l for l in seg if 'abort-probe i=' in l]
        def corrupt(l):
            m = re.search(r'badtok=(\d+)/', l)
            return int(m.group(1)) > 0 if m else float(re.search(r'ref_1mcos=([^ ]+)', l).group(1)) > 1e-3
        bad = [l for l in rows if corrupt(l)]
        r = dict(start=L[s][1:20], tier=re.search(r'档=(\S+)', L[s]).group(1), metal_hang_buffers=len(hang), metal_error_codes=dict(codes),
                 coreml_calls=len(rows), coreml_corrupted=len(bad), coreml_threw=sum(not re.search(r'threw=- ', l) for l in rows),
                 corrupted_without_abort_log=sum('oslog=-' in l for l in bad))
        for key in ('tailchk', 'tail2chk'):
            v = [(corrupt(l), float(m.group(1))) for l in rows for m in [re.search(key + r'=([^ /]+)', l)] if m]
            if v: r[key] = dict(flagged_corrupted=sum(c and x > 1e-4 for c, x in v), corrupted=sum(c for c, _ in v),
                                false_alarms=sum((not c) and x > 1e-4 for c, x in v), correct=sum(not c for c, _ in v))
        out[f'{f.split("/")[-2]} {r["start"]}'] = r; print(json.dumps(r, ensure_ascii=False))
json.dump(out, open('gpuhang_crosslayer.json', 'w'), ensure_ascii=False, indent=1)
