"""ICCE v28：实验 K 端到端（--gpu-check=tail）成片统计，只读 e2e/<机>_s<种子>_tail/trace.txt 与成片。
用法: python3 tail_e2e.py logs → tail_e2e.json"""
import sys, re, json, glob, subprocess, os, hashlib
R = sys.argv[1]; E = f'{R}/e2e'; out = {}
for d in sorted(glob.glob(f'{E}/*_tail')):
    name = os.path.basename(d); seed = re.search(r'_s(\d+)_', name).group(1)
    L = open(f'{d}/trace.txt', errors='ignore').read().splitlines()
    k = max(i for i, l in enumerate(L) if f'出片 seed={seed}' in l); L = L[k:]
    if not any('mp4 已写出' in l for l in L): continue
    tc = [l for l in L if '末尾校验 块' in l]; cos = [float(re.search(r'1-cos ([\d.e+-]+)', l).group(1)) for l in tc]
    ld = sum(float(re.search(r'加载 ([\d.]+)s', l).group(1)) for l in tc); rn = sum(float(re.search(r'推理 ([\d.]+)s', l).group(1)) for l in tc)
    blk = sum(float(m.group(1)) for l in L for m in [re.search(r'块\d\d 就位 [\d.]+s / 推理 ([\d.]+)s', l)] if m)
    total = max(float(m.group(1)) for l in L for m in [re.search(r'^\[\s*([\d.]+)s', l)] if m)
    ref = d.replace('_tail', '_log') + '/auto_output.mp4'
    # 公开包不含成片: 有 mp4 就照原法算整片 framemd5 的 md5, 没有就读包里预先导出的 auto_output.framemd5(同一条 ffmpeg 命令的输出)
    md5 = lambda p: hashlib.md5(open(p[:-4] + '.framemd5', 'rb').read()).hexdigest() if not os.path.exists(p) else subprocess.run(f'ffmpeg -v error -i {p} -f framemd5 - | md5sum', shell=True, capture_output=True, text=True).stdout.split()[0]
    out[name] = dict(tail_checks=len(tc), flagged=sum(c > 1e-4 for c in cos), flagged_min=min([c for c in cos if c > 1e-4], default=None),
                     ok_max=max([c for c in cos if c <= 1e-4], default=None), zero_reruns=sum('作废' in l and '精确 0' in l for l in L),
                     tail_load_s=round(ld), tail_run_s=round(rn), block_infer_s=round(blk), clip_s=round(total),
                     identical_to_log_arm=(os.path.exists(ref) or os.path.exists(ref[:-4] + '.framemd5')) and md5(f'{d}/auto_output.mp4') == md5(ref))
json.dump(out, open(f'{R}/tail_e2e.json', 'w'), indent=1); print(json.dumps(out, indent=1))
