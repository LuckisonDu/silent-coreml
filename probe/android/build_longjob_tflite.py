"""ICCE 实验 V2: 造「单个 GPU 任务很长」的 LiteRT 模型 = 一个 BATCH_MATMUL, X(1,1,M,K)·Y(1,1,K,N) → (1,1,M,N), 两个输入都是运行时张量。
配套探针 probe/icce_longjob_probe.cc(手机上按同一整数哈希现场生成 X、Y), 本脚本同时用 numpy 算参考输出 ref.f32 与 LiteRT CPU 对拍。
值 = ((lowbias32(i ^ seed·0x9E3779B9) % 17) - 8)/8: 1/8 的整数倍 ⇒ fp16 精确、fp32 累加精确, 参考 = 精确整数运算(int64)。
另有 --op conv: CONV_2D(VALID, stride 1), 输入 x(1,H,W,C) 运行时(种子 1), 权重 (C,k,k,C) 是模型常量(种子 2), 偏置 0;
  计算/字节比高 ⇒ 单个 kernel 很长而内存只有几十 MB(--M=H=W, --K=k, --N=C)。参考同样是精确整数运算(float32 BLAS 上的小整数, |和|·64 < 2^24)。
用法(B, venv logs/android/venv_build, 装了 ai-edge-litert):
  python build_longjob_tflite.py --M 2048 --K 16384 --N 2048 --out logs/android/lj/mm_2048_16384_2048
    产物 <out>.tflite <out>_ref.f32 <out>_info.json(含 GPU 兼容分析与 CPU 对拍)
"""
import argparse, hashlib, json, os, time
import numpy as np
from ai_edge_litert import schema_py_generated as S
import flatbuffers

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, required=True); ap.add_argument("--K", type=int, required=True); ap.add_argument("--N", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--op", choices=["matmul", "conv"], default="matmul")
ap.add_argument("--cpu-check", type=int, default=1, help="LiteRT CPU 跑一遍与 numpy 参考对拍")
a = ap.parse_args()
M, K, N = a.M, a.K, a.N
assert (K * K * N if a.op == "conv" else K) * 64 < 2 ** 24, "部分和要在 fp32 里精确"
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)


def gen(n, seed):  # 与 icce_longjob_probe.cc 的 gen() 逐位相同
    x = (np.arange(n, dtype=np.uint64).astype(np.uint32)) ^ np.uint32((seed * 0x9E3779B9) & 0xFFFFFFFF)
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16); x = (x * np.uint32(0x7feb352d)).astype(np.uint32)
    x ^= x >> np.uint32(15); x = (x * np.uint32(0x846ca68b)).astype(np.uint32)
    x ^= x >> np.uint32(16)
    return ((x % np.uint32(17)).astype(np.int64) - 8)   # 整数(×1/8 为实际值)



if a.op == "conv":
    H, k, C = M, K, N; Ho = H - k + 1
    xi = gen(H * H * C, 1).astype(np.float32).reshape(H, H, C); wi = gen(C * k * k * C, 2).astype(np.float32).reshape(C, k, k, C)
    m = S.ModelT(); m.version = 3; m.description = f"icce longjob conv {H}x{H}x{C} k{k}"
    oc = S.OperatorCodeT(); oc.builtinCode = oc.deprecatedBuiltinCode = S.BuiltinOperator.CONV_2D; oc.version = 1; m.operatorCodes = [oc]
    def buf(arr=None):
        bb = S.BufferT()
        if arr is not None: bb.data = np.frombuffer(np.ascontiguousarray(arr, np.float32).tobytes(), np.uint8)
        return bb
    m.buffers = [buf(), buf(), buf(wi / 8), buf(np.zeros(C, np.float32)), buf()]
    sg = S.SubGraphT(); sg.name = "main"
    def tens(name, shape, b_):
        t = S.TensorT(); t.name = name; t.shape = list(shape); t.type = S.TensorType.FLOAT32; t.buffer = b_; return t
    sg.tensors = [tens("x", (1, H, H, C), 1), tens("w", (C, k, k, C), 2), tens("b", (C,), 3), tens("out", (1, Ho, Ho, C), 4)]
    sg.inputs = [0]; sg.outputs = [3]
    op = S.OperatorT(); op.opcodeIndex = 0; op.inputs = [0, 1, 2]; op.outputs = [3]
    op.builtinOptionsType = S.BuiltinOptions.Conv2DOptions; o = S.Conv2DOptionsT(); o.padding = S.Padding.VALID
    o.strideW = o.strideH = 1; o.dilationWFactor = o.dilationHFactor = 1; o.fusedActivationFunction = S.ActivationFunctionType.NONE
    op.builtinOptions = o; sg.operators = [op]; m.subgraphs = [sg]
    b = flatbuffers.Builder(1 << 20); b.Finish(m.Pack(b), file_identifier=b"TFL3")
    fb = bytes(b.Output()); open(a.out + ".tflite", "wb").write(fb)
    t0 = time.time(); W2 = wi.transpose(1, 2, 3, 0).reshape(k * k * C, C)   # (k,k,Cin)→Cout
    ref = np.empty((Ho, Ho, C), np.float32)
    for r in range(Ho):   # 逐行 im2col(小整数在 float32 里精确)
        patch = np.stack([xi[r + dy, dx:dx + Ho, :] for dy in range(k) for dx in range(k)], 1).reshape(Ho, k * k * C)
        ref[r] = patch @ W2
    assert np.abs(ref).max() < 2 ** 24
    ref = ref / 64.0; ref.tofile(a.out + "_ref.f32")
    info = dict(op="conv", H=H, k=k, C=C, Ho=Ho, gflop=2 * Ho * Ho * C * C * k * k / 1e9, tflite_bytes=len(fb), tflite_sha256=hashlib.sha256(fb).hexdigest(),
                ref_sec=round(time.time() - t0, 1), ref_absmax=float(np.abs(ref).max()), ref_rms=float(np.sqrt((ref.astype(np.float64) ** 2).mean())),
                mem_mb=dict(x_fp32=H * H * C * 4 / 2**20, w_fp32=wi.size * 4 / 2**20, out_fp32=Ho * Ho * C * 4 / 2**20))
    from ai_edge_litert import _pywrap_analyzer_wrapper as an
    info["gpu_compat"] = [l for l in an.ModelAnalyzer(a.out + ".tflite", True, True).splitlines() if "ompatib" in l][:3]
    if a.cpu_check:
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=a.out + ".tflite", num_threads=8); it.allocate_tensors()
        it.set_tensor(it.get_input_details()[0]["index"], (xi / 8).reshape(1, H, H, C))
        t0 = time.time(); it.invoke(); info["cpu_sec"] = round(time.time() - t0, 2)
        y = it.get_tensor(it.get_output_details()[0]["index"]).reshape(Ho, Ho, C).astype(np.float64); rr = ref.astype(np.float64)
        info["cpu_vs_ref"] = dict(maxdiff=float(np.abs(y - rr).max()), one_minus_cos=float(1 - (y * rr).sum() / np.sqrt((y * y).sum() * (rr * rr).sum())))
    json.dump(info, open(a.out + "_info.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, ensure_ascii=False, indent=1)); raise SystemExit(0)

# ── 模型 ──
m = S.ModelT(); m.version = 3; m.description = f"icce longjob batch_matmul {M}x{K}x{N}"
m.operatorCodes = [S.OperatorCodeT()]
m.operatorCodes[0].builtinCode = S.BuiltinOperator.BATCH_MATMUL
m.operatorCodes[0].deprecatedBuiltinCode = S.BuiltinOperator.BATCH_MATMUL; m.operatorCodes[0].version = 1
m.buffers = [S.BufferT() for _ in range(4)]
sg = S.SubGraphT(); sg.name = "main"
def tens(name, shape, buf):
    t = S.TensorT(); t.name = name; t.shape = list(shape); t.type = S.TensorType.FLOAT32; t.buffer = buf; return t
sg.tensors = [tens("x", (1, 1, M, K), 1), tens("y", (1, 1, K, N), 2), tens("out", (1, 1, M, N), 3)]
sg.inputs = [0, 1]; sg.outputs = [2]
op = S.OperatorT(); op.opcodeIndex = 0; op.inputs = [0, 1]; op.outputs = [2]
op.builtinOptionsType = S.BuiltinOptions.BatchMatMulOptions
op.builtinOptions = S.BatchMatMulOptionsT(); op.builtinOptions.adjX = False; op.builtinOptions.adjY = False
sg.operators = [op]
m.subgraphs = [sg]
b = flatbuffers.Builder(1024); b.Finish(m.Pack(b), file_identifier=b"TFL3")
fb = bytes(b.Output()); open(a.out + ".tflite", "wb").write(fb)

# ── 参考(精确): ref = (Xi @ Yi) / 64 ──
t0 = time.time()
Xi = gen(M * K, 1).reshape(M, K).astype(np.float64); Yi = gen(K * N, 2).reshape(K, N).astype(np.float64)
ref = (Xi @ Yi) / 64.0      # 整数在 float64 里精确(|和| < 2^53)
assert np.abs(ref).max() * 64 < 2 ** 24
ref.astype(np.float32).tofile(a.out + "_ref.f32")
info = dict(M=M, K=K, N=N, gflop=2 * M * K * N / 1e9, tflite_bytes=len(fb), tflite_sha256=hashlib.sha256(fb).hexdigest(),
            ref_sec=round(time.time() - t0, 1), ref_absmax=float(np.abs(ref).max()), ref_rms=float(np.sqrt((ref ** 2).mean())),
            mem_mb=dict(x_fp32=M * K * 4 / 2**20, y_fp32=K * N * 4 / 2**20, out_fp32=M * N * 4 / 2**20))
from ai_edge_litert import _pywrap_analyzer_wrapper as an
rep = an.ModelAnalyzer(a.out + ".tflite", True, True)
info["gpu_compat"] = [l for l in rep.splitlines() if "ompatib" in l or "GPU" in l][:6]
if a.cpu_check:
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=a.out + ".tflite", num_threads=8); it.allocate_tensors()
    d = it.get_input_details()
    it.set_tensor(d[0]["index"], (Xi / 8).astype(np.float32).reshape(1, 1, M, K)); it.set_tensor(d[1]["index"], (Yi / 8).astype(np.float32).reshape(1, 1, K, N))
    t0 = time.time(); it.invoke(); info["cpu_sec"] = round(time.time() - t0, 2)
    y = it.get_tensor(it.get_output_details()[0]["index"]).reshape(M, N).astype(np.float64)
    info["cpu_vs_ref"] = dict(maxdiff=float(np.abs(y - ref).max()), one_minus_cos=float(1 - (y * ref).sum() / np.sqrt((y * y).sum() * (ref * ref).sum())))
json.dump(info, open(a.out + "_info.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(info, ensure_ascii=False, indent=1))
