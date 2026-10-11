"""同种子三组成片两两比（同种子确定性 ⇒ 干净的片应逐帧相同）。写 <e2e>/e2e_compare.json。
有 auto_output.mp4 时按原版逐帧 PSNR（缩到 352x640）；公开包只带 auto_output.framemd5（ffmpeg -f framemd5 的输出），
此时只判逐帧是否相同（identical），PSNR 记 None —— 论文用到的是 identical（"pixel-identical between arms"）。
用法: python3 e2e_compare.py logs/e2e"""
import subprocess, glob, os, itertools, json, sys
root = sys.argv[1]; res = []
def frames(p):
    import numpy as np
    r = subprocess.run(['ffmpeg','-v','error','-i',p,'-vf','scale=352:640','-f','rawvideo','-pix_fmt','rgb24','-'], capture_output=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, 640, 352, 3).astype(np.float32)
def fmd5(p): return [l.split(',')[-1].strip() for l in open(p) if l.strip() and not l.startswith('#')]
seeds = sorted({os.path.basename(d).rsplit('_', 1)[0] for d in glob.glob(f'{root}/*_s*_*') if '_failed' not in d})
for s in seeds:
    arms = [a for a in ['none', 'zero', 'log'] if os.path.exists(f'{root}/{s}_{a}/auto_output.mp4') or os.path.exists(f'{root}/{s}_{a}/auto_output.framemd5')]
    for a, b in itertools.combinations(arms, 2):
        pa, pb = f'{root}/{s}_{a}/auto_output.mp4', f'{root}/{s}_{b}/auto_output.mp4'
        if os.path.exists(pa) and os.path.exists(pb):
            import numpy as np
            A, B = frames(pa), frames(pb); n = min(len(A), len(B)); mse = ((A[:n] - B[:n]) ** 2).mean(axis=(1, 2, 3))
            p = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-9))
            r = dict(seed=s, a=a, b=b, psnr_median=round(float(np.median(p)), 1), psnr_min=round(float(p.min()), 1), identical=bool(mse.max() == 0))
        else:
            ha, hb = fmd5(pa[:-4] + '.framemd5'), fmd5(pb[:-4] + '.framemd5')
            r = dict(seed=s, a=a, b=b, psnr_median=None, psnr_min=None, identical=ha == hb)
        res.append(r); print(r)
json.dump(res, open(f'{root}/e2e_compare.json', 'w'), indent=1)
