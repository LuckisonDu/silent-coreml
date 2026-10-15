"""ICCE 10-13: 算「第 0 块单独」的 torch fp32 参考输出 blk00_out.f16(用 Pexels 输入), 并先对旧 dump 做校验。
在 0.4 CPU 上跑: /data/icce_chain8/venv/bin/python pexels_block0_ref.py   (同目录要有 export_chain_blocks.py 与 export_all_modcompact.py 的拷贝)
块前向/权重加载与 export_chain_blocks.py 同源(直接 exec 它到 def main() 之前的部分), 只做 torch 参考, 不导出 Core ML。
参数硬写在这里(C1180), 不用命令行。"""
import os, sys, json
# ===== 在这里改 =====
CODE_DIR  = "/data/icce_pexels_blk0/code"                  # export_chain_blocks.py / export_all_modcompact.py 的拷贝所在
MODEL_DIR = "/data/turbo/Wan2.2-TI2V-5B-Turbo"             # Turbo 权重(bf16)
OLD_DUMP  = "/data/icce_pexels_blk0/old_dump"              # 旧输入(仅内部校验用, 不公开): stem_out ctx mod3 rope blk00_out
PEX_DUMP  = "/data/icce_pexels_blk0/pexels_in"             # Pexels 输入: stem_out ctx mod3 rope
OUT_DIR   = "/data/icce_pexels_blk0/out"                   # 输出: pexels_blk00_out.f16, old_recompute_blk00_out.f16, check.json
# ====================
os.environ.update(FW_CHAIN="0", FW_MODEL=MODEL_DIR, FW_OUT=OUT_DIR + "/_unused", FW_VERIFY="0", FW_DUMP=OLD_DUMP, FW_ORIG=OUT_DIR + "/_none")
os.makedirs(OUT_DIR, exist_ok=True)
src_path = CODE_DIR + "/export_chain_blocks.py"
src = open(src_path).read()
g = {"__name__": "ref_defs", "__file__": src_path}
exec(compile(src[:src.index("def main():")], src_path, "exec"), g)
torch, np, T = g["torch"], g["NS"]["np"], g["T"]
blk = g["make_block"](0)
g["NS"]["tr"].blocks = None
blk.requires_grad_(False)
rd = lambda d, n, sh: torch.from_numpy(np.fromfile(f"{d}/{n}.f16", np.float16).reshape(sh).copy())
def ref(d):
    a = rd(d, "stem_out", (1, T, 3072)).float(); b = rd(d, "ctx", (1, 512, 3072)).float()
    c = rd(d, "mod3", (1, 3, 6, 3072)).float(); r = rd(d, "rope", (1, 2, T, 64)).float()
    with torch.no_grad():
        return blk(a, b, c, r)
res = {}
y_old = ref(OLD_DUMP)
y_old.half().numpy().tofile(OUT_DIR + "/old_recompute_blk00_out.f16")
old_ref = rd(OLD_DUMP, "blk00_out", (1, T, 3072))
res["old_recompute_vs_old_blk00_out"] = g["_cmp"](y_old.half().float(), old_ref.float())
res["old_recompute_fp32_vs_old_blk00_out"] = g["_cmp"](y_old, old_ref.float())
res["old_stats"] = dict(rms=float(y_old.pow(2).mean().sqrt()), absmax=float(y_old.abs().max()))
print(res, flush=True); json.dump(res, open(OUT_DIR + "/check.json", "w"), indent=1, ensure_ascii=False)
y = ref(PEX_DUMP)
y.half().numpy().tofile(OUT_DIR + "/pexels_blk00_out.f16")
res["pexels_stats"] = dict(rms=float(y.pow(2).mean().sqrt()), absmax=float(y.abs().max()), finite=bool(torch.isfinite(y).all()))
res["pexels_fp16_roundtrip"] = g["_cmp"](y.half().float(), y)
print(res, flush=True); json.dump(res, open(OUT_DIR + "/check.json", "w"), indent=1, ensure_ascii=False)
