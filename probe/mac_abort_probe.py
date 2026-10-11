"""ICCE 新题 A 的 Mac(M4) 对照：同一个 block00 包、同一份手机落盘输入，在 Mac 的 CoreML GPU 上连续推理，
逐次记耗时、输出统计、对参考的余弦，并用 `log show` 查同一时间窗里的 GPU 命令缓冲掐断记录。
用法: python3 mac_abort_probe.py <block00.mlpackage> <dump目录> <次数> [cpu_and_gpu|all]"""
import sys, time, subprocess, datetime, json
import numpy as np, coremltools as ct
pkg, dump, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); cu = sys.argv[4] if len(sys.argv) > 4 else 'cpu_and_gpu'
f16 = lambda k: np.fromfile(f'{dump}/{k}.f16', dtype=np.float16)
dim = 3072; rope = f16('rope'); T = rope.size // (2 * 64)
feed = {'a': f16('stem_out').reshape(1, T, dim), 'b': f16('ctx').reshape(1, -1, dim), 'c': f16('mod3').astype(np.float32).reshape(1, 3, 6, dim), 'd': rope.reshape(1, 2, T, 64)}
ref = f16('blk00_out').astype(np.float32).ravel()
m = ct.models.MLModel(pkg, compute_units={'cpu_and_gpu': ct.ComputeUnit.CPU_AND_GPU, 'all': ct.ComputeUnit.ALL}[cu])
names = [i.name for i in m.get_spec().description.input]; feed = {k: v for k, v in feed.items() if k in names}
print('T', T, 'inputs', names, flush=True)
for i in range(n):
    t0 = datetime.datetime.now(); s = time.time(); err = '-'
    try: out = list(m.predict(feed).values())[0].astype(np.float32).ravel()
    except Exception as e: out = None; err = str(e)[:120]
    dt = time.time() - s; t1 = datetime.datetime.now()
    lg = subprocess.run(['log', 'show', '--start', t0.strftime('%Y-%m-%d %H:%M:%S'), '--end', (t1 + datetime.timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S'),
                         '--predicate', 'eventMessage CONTAINS "kIOGPUCommandBufferCallbackError" OR eventMessage CONTAINS "Impacting Interactivity" OR eventMessage CONTAINS "InnocentVictim"', '--style', 'compact'],
                        capture_output=True, text=True).stdout
    ab = sum(1 for l in lg.splitlines() if 'Error' in l or 'Impacting' in l or 'Innocent' in l)
    if out is not None:
        c = float(out @ ref / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
        print(json.dumps({'i': i, 'dt': round(dt, 2), 'err': err, 'abort_lines': ab, 'zero': float((out == 0).mean()), 'nan': int((~np.isfinite(out)).sum()), 'rms': float(np.sqrt(np.mean(out ** 2))), 'cos': c}), flush=True)
    else:
        print(json.dumps({'i': i, 'dt': round(dt, 2), 'err': err, 'abort_lines': ab}), flush=True)
