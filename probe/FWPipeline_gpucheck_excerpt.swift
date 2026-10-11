// Excerpt (not compilable on its own) from ios_work/Sources/FWPipeline.swift of the app source tree,
// i.e. the production per-block check used for the full clips (--gpu-check=log|zero|none|tail|stride).
// Only the parts relevant to the paper are kept; line numbers refer to the original file as of 2026-10-10.
// 摘自 App 源码 ios_work/Sources/FWPipeline.swift（产线逐块检查 --gpu-check=…），只摘与论文有关的部分，不能单独编译。

// ---- [FWPipeline.swift L1426-1480] retry waits, check mode, tailCheck() (tail = last K rows, stride = p_k = floor((k+0.5)T/K)) ----
    /// 第 n 次重算前等多久(秒), 逐级加长; 次数 = 数组长度。
    /// ★不看发热状态: 09-13 验收轮里它一直停在「严重/危急」, 等满 180s 也降不下来, 但重算照样一次成功。
    ///   宁可慢也不交坏片、不中途失败 ⇒ 给足 8 次。
    nonisolated(unsafe) public static var retryWaits: [Double] = [10, 30, 60, 120, 180, 240, 300, 300]
    /// 被掐后接下来几块开算前歇多久(见块循环「块前歇」注释)。abortCooldownBlocks=0 关闭。
    nonisolated(unsafe) public static var abortCooldownBlocks = 4
    /// ★块级 GPU 检查方式(10-05 ICCE 端到端对照加, `--gpu-check=`): log = 现役(日志判据 + 全零/NaN, 被掐就回退重算);
    ///   zero = 只验全零/NaN(日志仍在后台读, 只记录不处理); none = 什么都不验(同上只记录)。默认 log, 产线不变。
    nonisolated(unsafe) public static var gpuCheckMode = "log"
    /// ★末尾校验(10-07 ICCE 实验 K, `--gpu-check=tail`): 同档旁边的 `<档>t64/blockNN_int8.mlpackage` 只算最后 64 个 token 的块输出,
    ///   与整块输出末尾 64 行比; 1−cos > tailThreshold 判坏, 走 validated 的「等待后重算」。不读系统日志(日志仍在后台读, 只记录)。
    ///   每块用时加载、用完即放(随 autoreleasepool), 记加载与推理耗时。校验包缺失或推理报错 ⇒ 判坏(宁可多重算)。
    nonisolated(unsafe) public static var tailThreshold = 1e-4
    nonisolated(unsafe) static var tailStats: (calls: Int, flagged: Int, load: Double, run: Double) = (0, 0, 0, 0)
    static func tailCheck(tierDir: URL, block: Int, feed: [String: MLMultiArray], out: MLMultiArray,
                          units: MLComputeUnits, cacheDir: URL?) -> String? {
        // ★10-09 --gpu-check=stride(ICCE v59: 分散检查当主方法): 校验包在 <档>s64/(same block weights, exported to output only rows p_k),
        //   输出第 k 行对应整块第 p_k = floor((k+0.5)·T/K) 行(与 paper_icce/silent/analysis/check_positions.py 同式); tail 仍是 <档>t64/ 的最后 K 行。
        let strided = gpuCheckMode == "stride"
        let tdir = tierDir.deletingLastPathComponent().appendingPathComponent(tierDir.lastPathComponent + (strided ? "s64" : "t64"), isDirectory: true)
        let url = tdir.appendingPathComponent(String(format: "block%02d_int8.mlpackage", block))
        let t0 = Date()
        return autoreleasepool { () -> String? in
            tailStats.calls += 1
            guard FileManager.default.fileExists(atPath: url.path) else { tailStats.flagged += 1; return "末尾校验包缺失 \(url.lastPathComponent)" }
            do {
                let m = try CompiledModel(packageURL: url, computeUnits: units, cacheDir: cacheDir)   // 与整块同一个缓存目录 ⇒ 编译产物的 weight.bin 与整块同 sha, 被 dedupeCompiledWeight 硬链, 不多占盘
                let tl = -t0.timeIntervalSinceNow
                let t1 = Date()
                let o = try m.predict(feed)
                guard let v = o.values.first else { tailStats.flagged += 1; return "末尾校验无输出" }
                let tv = try detach(v)
                let tr = -t1.timeIntervalSinceNow
                tailStats.load += tl; tailStats.run += tr
                let tb = tv.fp16Buffer, ob = out.fp16Buffer, dim = 3072
                let K = tb.count / dim, T = ob.count / dim, off = (T - K) * dim
                var dot = 0.0, na = 0.0, nb = 0.0
                if strided {
                    for k in 0..<K {
                        let p = Int((Double(k) + 0.5) * Double(T) / Double(K))
                        for c in 0..<dim { let x = Double(ob[p * dim + c]), y = Double(tb[k * dim + c]); dot += x * y; na += x * x; nb += y * y }
                    }
                } else {
                    for j in 0..<tb.count { let x = Double(ob[off + j]), y = Double(tb[j]); dot += x * y; na += x * x; nb += y * y }
                }
                let e = 1 - dot / max((na * nb).squareRoot(), 1e-30)
                Trace.mark(String(format: "  %@校验 块%02d 1-cos %.2e 加载 %.2fs 推理 %.2fs", strided ? "分散" : "末尾", block, e, tl, tr))
                if e > tailThreshold || !e.isFinite { tailStats.flagged += 1; return String(format: "末尾校验不符 1-cos %.2e", e) }
                return nil
            } catch {
                tailStats.flagged += 1
                return "末尾校验报错 \(String(describing: error).prefix(80))"
            }
        }
    }

// ---- [FWPipeline.swift L997-1012] call site inside the block loop: verify each block output before the next block uses it ----
                // 先收上一块的掐断结论: 被掐过就丢掉这一块刚算的结果, 回退重算
                if try takePending() { continue }
                // ★末尾校验放在块的 autoreleasepool 之外(10-07 实验 K 首轮: 整块与校验模型同时在内存里, 两台手机在第 2 块后 App 静默退出、无 jetsam 记录)
                if Self.gpuCheckMode == "tail" || Self.gpuCheckMode == "stride",
                   let bad = Self.tailCheck(tierDir: st.dir, block: i, feed: tailFeed, out: next, units: computeUnits, cacheDir: cacheDir) {
                    tailAttempts[i] += 1
                    let n = tailAttempts[i]
                    guard n <= Self.retryWaits.count else {
                        throw LTXError.missingOutput("第\(step + 1)步第\(i + 1)块 末尾校验连续 \(n) 次不符(\(bad)) —— 请让手机凉下来再试")
                    }
                    let wait = Self.retryWaits[n - 1]
                    Trace.mark(String(format: "  ⚠️ 第%d步第%d块 作废(%@, 发热 %@) —— 等 %.0fs 后重算 %d/%d", step + 1, i + 1, bad, Self.thermalText, wait, n, Self.retryWaits.count))
                    Thread.sleep(forTimeInterval: wait)
                    hidden = inputOfThis
                    continue
                }

// ---- [FWPipeline.swift L1830-1836] batch size of the GPU-abort log check ----
    /// ★10-08 GPU 掐断日志成批查的批大小（见块循环 batchStart）。--gpu-check-batch=N 强制（N=1 = 逐块，论文复现用）；
    ///   不给: 新机（bigPhoneFastPath）10，其余 1（iPhone 12 / --auto 不变）。
    static let gpuCheckBatch: Int = {
        if let a = CommandLine.arguments.first(where: { $0.hasPrefix("--gpu-check-batch=") }),
           let n = Int(a.dropFirst("--gpu-check-batch=".count)), n >= 1 { return n }
        return bigPhoneFastPath ? 10 : 1
    }()
