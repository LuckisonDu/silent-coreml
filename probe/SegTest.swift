import Foundation
import CoreML
import OSLog
import Metal
#if canImport(UIKit)
import UIKit
#endif

/// ICCE 2027 companion: on-device probe for silent Core ML output corruption (excerpt of the app's SegTest.swift: abortProbe and its helpers only).
enum SegTest {
    /// ★--abort-probe=<档>（ICCE 新题 A，2026-10-05）：同一个块（block00）在**同一份真实输入**上连续推理 N 次，把手机烧热，
    ///   逐次记下 iOS GPU 看门狗把命令缓冲掐掉时**运行时实际交回了什么**——用来判断「静默失败」是否存在、简单检查抓不抓得住。
    ///   · 直接调 `model.prediction`（**不走** CompiledModel.predict 的自动重试），抛没抛错原样记下。
    ///   · 每次一行（立刻写盘，被杀也留得下）：序号、发热、耗时、抛错文字、本次时间窗里 os_log 的掐断记录、
    ///     输出里恰为 0 的比例、NaN/Inf 个数、均方根、对参考的 1-cos 与最大差。
    ///   · 参考 = 手机当时落盘的 fw_dump/blk00_out.f16（09-13 已在 Mac 上核对 cos 1.000000）；另记对本档第 1 次（凉机）输出的 1-cos。
    ///   · 输出缓冲由 CoreML 分配（不传 outputBackings）⇒ 全零若出现，就是运行时交回的，不是 App 没写。
    ///   结果写 Documents/abort_probe.txt；--abort-probe-iters=N（默认 200）。
    /// 每个 token 的 L2 范数（fp16 [1,T,dim]）。
    /// ★GPU 测速(10-06, ICCE 机制实验): iOS 不给读 GPU 频率 ⇒ 跑一个固定工作量的小 Metal 核(每线程 N 次乘加),
    ///   取 GPU 自报时长(gpuEndTime − gpuStartTime, 毫秒)当「当前 GPU 速度」。同一台机器上数越大 = GPU 越慢(降频 / 低性能状态)。

    /// --abort-probe-gpuhang 用的超长循环核(同步建库; async 上下文里 makeLibrary 会选到 async 重载)
    static func makeHangKernel() -> (MTLCommandQueue, MTLComputePipelineState, MTLBuffer)? {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void hang(device float *o [[buffer(0)]], constant uint &n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            float a = float(i);
            for (uint k = 0; k < n; k++) { a = fma(a, 1.0000001f, 1e-7f); }
            o[i] = a;
        }
        """
        guard let lib = try? d.makeLibrary(source: src, options: nil), let fn = lib.makeFunction(name: "hang"),
              let pso = try? d.makeComputePipelineState(function: fn), let buf = d.makeBuffer(length: 4096 * 4) else { return nil }
        return (q, pso, buf)
    }
    struct UncheckedBox<T>: @unchecked Sendable { let v: T; init(_ v: T) { self.v = v } }
    // ★10-09 --abort-probe-metalsent: 后台每 0.2 s 提交一个极短的 Metal 小核(约 1 ms, 不占住 GPU、不诱发),
    //   记每个命令缓冲的完成状态。自然发热时 iOS 掐断 GPU ⇒ 看同一时间窗里我们自己的 Metal 缓冲有没有拿到错误码(API 层可见), 而 Core ML predict 照常返回。
    final class MetalSentinel: @unchecked Sendable {
        private let l = NSLock(); private var ev: [(Date, Int)] = []   // (完成时刻, 错误码; 0 = 正常)
        func add(_ c: Int) { l.lock(); ev.append((Date(), c)); l.unlock() }
        func window(_ a: Date, _ b: Date) -> String {
            l.lock(); let w = ev.filter { $0.0 >= a && $0.0 <= b }; l.unlock()
            let e = w.filter { $0.1 != 0 }
            let codes = Dictionary(grouping: e, by: { $0.1 }).map { "\($0.key)x\($0.value.count)" }.sorted().joined(separator: ",")
            return "msent=\(e.count)/\(w.count)" + (codes.isEmpty ? "" : " mcodes=\(codes)")
        }
    }
    final class GPUSpeedo {
        let dev: MTLDevice, q: MTLCommandQueue, pso: MTLComputePipelineState, buf: MTLBuffer
        init?() {
            guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void spin(device float *o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
                float a = float(i) * 1e-6f, b = 1.0001f;
                for (int k = 0; k < 4096; k++) { a = fma(a, b, 1e-7f); b = fma(b, 0.99999f, 1e-6f); }
                o[i] = a + b;
            }
            """
            guard let lib = try? d.makeLibrary(source: src, options: nil), let fn = lib.makeFunction(name: "spin"),
                  let p = try? d.makeComputePipelineState(function: fn), let b = d.makeBuffer(length: 65536 * 4) else { return nil }
            dev = d; self.q = q; pso = p; buf = b
        }
        /// 跑一次, 返回 GPU 时长(ms); 失败返回 -1
        func measure() -> Double {
            guard let cb = q.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return -1 }
            enc.setComputePipelineState(pso); enc.setBuffer(buf, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: 65536, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return cb.error == nil ? (cb.gpuEndTime - cb.gpuStartTime) * 1000 : -1
        }
        /// 「保温」: 在 seconds 内反复跑小核, 不让 GPU 空闲
        func keepWarm(_ seconds: Double) { let t0 = Date(); while -t0.timeIntervalSinceNow < seconds { _ = measure() } }
    }

    static func tokenNorms(_ x: MLMultiArray, T: Int, dim: Int) -> [Double] {
        let b = x.fp16Buffer; var out = [Double](repeating: 0, count: T)
        for t in 0..<T { var s = 0.0; let o = t * dim; for k in 0..<dim { let v = Double(b[o + k]); s += v * v }; out[t] = s.squareRoot() }
        return out
    }

    /// 本进程 GPU 相关错误日志（带时间戳），用于按时间回填。
    static func gpuLogsWithDates(since: Date) -> [(Date, String)] {
        guard let store = try? OSLogStore(scope: .currentProcessIdentifier) else { return [] }
        let pred = NSPredicate(format: "composedMessage CONTAINS[c] %@", "kIOGPUCommandBufferCallbackError")
        guard let entries = try? store.getEntries(with: [], at: store.position(date: since), matching: pred) else { return [] }
        var out: [(Date, String)] = []
        for e in entries { guard e.date >= since, let le = e as? OSLogEntryLog else { continue }; out.append((le.date, String(le.composedMessage.prefix(200)))) }
        return out
    }

    static func abortProbe(tier: String, iters: Int, dumpSub: String = "fw_dump", recover: [String] = [],
                           note: @escaping (String) -> Void) async {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dump = docs.appendingPathComponent(dumpSub, isDirectory: true)
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("abortprobe", isDirectory: true)
        let logURL = docs.appendingPathComponent("abort_probe.txt")
        func say(_ t: String) {
            let line = "[\(Date())] \(t)\n"
            if let h = try? FileHandle(forWritingTo: logURL) { h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close() }
            else { try? Data(line.utf8).write(to: logURL, options: .atomic) }
            note(t)
        }
        func th() -> Int { ProcessInfo.processInfo.thermalState.rawValue }
        // --abort-probe-minbat=百分比: 电量低于它就停(默认 30; 放电扫描要跑到 15)
        let thermalGate = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-thermalgate=") }).flatMap { Double($0.dropFirst("--abort-probe-thermalgate=".count)) } ?? 0
        // ★--abort-probe-sentinel(10-08 ICCE 内部审稿意见「和更便宜的检测比」): 每次调用前自己分配输出缓冲, 整块填 NaN,
        //   经 MLPredictionOptions.outputBackings 交给 Core ML 写。若被掐断时后半段根本没写进来, 结果里会剩 NaN ⇒ 零成本检测;
        //   若 Core ML 先在自己的 GPU 缓冲里算完再整块拷进来, 哨兵会被旧值盖掉 ⇒ 这个办法不行(也是要报告的结果)。
        //   日志: sent=未被改写(仍是 NaN)的 token 数 / 首个仍是 NaN 的 token; naninf= 照旧统计。
        let sentinel = CommandLine.arguments.contains("--abort-probe-sentinel")
        let sentFresh = CommandLine.arguments.contains("--abort-probe-sentinel-fresh")
        let noSpeedo = CommandLine.arguments.contains("--abort-probe-nospeedo")
        let cpuHeatMax = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-cpuheat=") }).flatMap { Double($0.dropFirst("--abort-probe-cpuheat=".count)) } ?? 0
        let minBat = Float(CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-minbat=") }).flatMap { Double($0.dropFirst("--abort-probe-minbat=".count)) } ?? 30) / 100
        let dim = 3072
        do {
            func load(_ n: String) throws -> Data { try Data(contentsOf: dump.appendingPathComponent(n + ".f16")) }
            let ropeD = try load("rope"), ctxD = try load("ctx"), modD = try load("mod3"), stemD = try load("stem_out"), refD = try load("blk00_out")
            let T = ropeD.count / (2 * 64 * 2)
            let rope = try MLMultiArray.fp16(shape: [1, 2, T, 64]); rope.fill(fromFP16: ropeD)
            let ctx = try MLMultiArray.fp16(shape: [1, ctxD.count / (2 * dim), dim]); ctx.fill(fromFP16: ctxD)
            let mod = try MLMultiArray(shape: [1, 3, 6, dim].map { NSNumber(value: $0) }, dataType: .float32)
            modD.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Float16.self); let dst = mod.dataPointer.bindMemory(to: Float.self, capacity: mod.count)
                for i in 0..<min(src.count, mod.count) { dst[i] = Float(src[i]) }
            }
            let a = try MLMultiArray.fp16(shape: [1, T, dim]); a.fill(fromFP16: stemD)
            let ref = try MLMultiArray.fp16(shape: [1, T, dim]); ref.fill(fromFP16: refD)
            // ★--abort-probe-units=cpu(10-07 实验 L, 外审要「排除 Core ML 运行时本身」): 同一模型同一输入改走 CPU, 不受 GPU 看门狗管 ⇒ 若热机仍不错, 问题在 GPU 那一步
            let unitsArg = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-units=") }).map { String($0.dropFirst("--abort-probe-units=".count)) } ?? "gpu"
            let probeUnits: MLComputeUnits = unitsArg == "cpu" ? .cpuOnly : .cpuAndGPU
            func loadModel() throws -> CompiledModel {
                try CompiledModel(packageURL: FWStore(tier: tier).root.appendingPathComponent("block00_int8.mlpackage"),
                                  computeUnits: probeUnits, cacheDir: caches.appendingPathComponent(unitsArg == "cpu" ? tier + "_cpu" : tier, isDirectory: true))
            }
            // ★拆块预防实验(10-06): 档目录里有 block00a/b(/c)_int8 ⇒ 一块分几段连着调用(段间不停、不读日志, 免得 GPU 歇一下混进对比),
            //   段输出按名字接到下一段输入(h1 / h / o, 见 fastwan/export/export_split_block.py); 每段起止记下, 跑完按时间戳回填日志。
            let root = FWStore(tier: tier).root
            let stageFiles = ["block00a_int8", "block00b_int8", "block00c_int8"]
                .filter { FileManager.default.fileExists(atPath: root.appendingPathComponent($0 + ".mlpackage").path) }
            // ★段模型不常驻(10-06 真机: 三个同时加载, 段0→段2 可用内存 1388→287 MB 被杀) ⇒ 每段用时加载、用完释放; 先各加载一次把编译缓存建好。
            func loadStage(_ f: String) throws -> CompiledModel {
                try CompiledModel(packageURL: root.appendingPathComponent(f + ".mlpackage"), computeUnits: .cpuAndGPU,
                                  cacheDir: caches.appendingPathComponent(tier, isDirectory: true))
            }
            for f in stageFiles { try autoreleasepool { _ = try loadStage(f) } }
            let stages = stageFiles
            var loadDts: [Double] = []
            var stageWins: [(Int, Int, Date, Date)] = []
            let speedo = GPUSpeedo()
            var segSpeed: [Double] = []
            if speedo == nil { say("⚠️ GPU 测速核建不起来, 不记 gpuspd") }
            /// 依次跑各段; 返回最后一段输出与每段耗时
            // --abort-probe-only=k: 只跑第 k 段(诊断真机卡住; 缺的中间输入 h1/h 用原始 a 凑形状, 结果不对, 只看能否算完与耗时)
            let only = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-only=") }).flatMap { Int($0.dropFirst("--abort-probe-only=".count)) }
            // --abort-probe-gap=秒: 段与段之间额外让 GPU 空着(检验「段间空隙让系统插进来」是不是拆段有效的原因)
            let gap = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-gap=") }).flatMap { Double($0.dropFirst("--abort-probe-gap=".count)) } ?? 0
            // --abort-probe-gapwarm: 段间那段时间不让 GPU 空着, 反复跑测速小核「保温」(检验「空闲后 GPU 降到低性能状态」)
            let gapWarm = CommandLine.arguments.contains("--abort-probe-gapwarm")
            if gap > 0 { say("abort-probe 段间额外停 \(gap) s\(gapWarm ? " (保温)" : "")") }
            func runChain(_ i: Int) throws -> (MLMultiArray?, [Double]) {
                var pool: [String: MLMultiArray] = ["a": a, "b": ctx, "c": mod, "d": rope]
                if only != nil {
                    pool["h"] = try MLMultiArray(shape: [1, NSNumber(value: T), NSNumber(value: dim)], dataType: .float32)
                    let h1 = try MLMultiArray(shape: [1, NSNumber(value: T / 2), NSNumber(value: dim)], dataType: .float32)
                    pool["h1"] = h1
                }
                var last: MLMultiArray?, dts: [Double] = []
                loadDts = []
                for (k, f) in stages.enumerated() where only == nil || only == k {
                  try autoreleasepool {
                    let l0 = Date()
                    let st = try loadStage(f)
                    loadDts.append(-l0.timeIntervalSinceNow)
                    let want = Set(st.model.modelDescription.inputDescriptionsByName.keys)
                    let pv = try MLDictionaryFeatureProvider(dictionary: pool.filter { want.contains($0.key) }.mapValues { MLFeatureValue(multiArray: $0) })
                    if i < 2 { say("  分段调用 i=\(i) 段\(k) 开始 可用内存 \(FWPipeline.availableMemoryMB()) MB") }   // 前两次留面包屑: 再静默消失时知道停在哪段
                    let s0 = Date()
                    let out = try st.model.prediction(from: pv)
                    // 输出拷成自有数组再放手, 让模型连同它的中间缓冲一起释放
                    for n in out.featureNames { if let v = out.featureValue(for: n)?.multiArrayValue {
                        let c = try MLMultiArray(shape: v.shape, dataType: v.dataType)
                        v.withUnsafeBytes { src in c.withUnsafeMutableBytes { dst, _ in dst.copyMemory(from: src) } }
                        pool[n] = c; last = c } }
                    stageWins.append((i, k, s0, Date())); dts.append(-s0.timeIntervalSinceNow)
                    if gap > 0 && k < stages.count - 1 && only == nil {
                        if gapWarm, let sp = speedo { sp.keepWarm(gap) } else { Thread.sleep(forTimeInterval: gap) }
                    }
                    if k < stages.count - 1 && only == nil, let sp = speedo { segSpeed.append(sp.measure()) }
                  }
                }
                return (last, dts)
            }
            if !stages.isEmpty { say("abort-probe 分段: \(stageFiles.joined(separator: " → "))") }
            // --abort-probe-plan: 只报各段(或整块)每个算子的计划设备(MLComputePlan), 不预测 —— 查三段档首段真机卡住是否退回 CPU
            if CommandLine.arguments.contains("--abort-probe-plan") {
                let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
                for f in (stageFiles.isEmpty ? ["block00_int8"] : stageFiles) {
                    await computePlanReport("\(tier)/\(f)", compiled: caches.appendingPathComponent(tier).appendingPathComponent(FWStore.compiledCacheName(for: root.appendingPathComponent(f + ".mlpackage")) + ".mlmodelc"), cfg: cfg, say: say)
                }
                say("✅ abort-probe 完成(只报计划)")
                return
            }
            // 单块档才加载整块(分段档目录里没有 block00_int8)
            var m: CompiledModel! = stages.isEmpty ? try loadModel() : nil
            // ★--abort-probe-tail=<档>: 末尾校验(不靠日志的检测, 10-06)。被掐的输出都是「从某处一直坏到最后一个 token」⇒
            //   另跑一个只出最后 K 个 token 的同权重模型(fastwan/export/export_split_block.py FW_SPLIT_AT=tail), 与整块输出末尾 K 行比对。
            let tailTier = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-tail=") }).map { String($0.dropFirst("--abort-probe-tail=".count)) }
            // ★两个 fp32 模型同时常驻 ⇒ 10-06 phoneA 首次调用即被杀(同三段常驻那次) ⇒ 校验模型用时加载、用完释放, 校验前也先放掉整块模型
            func loadTail() throws -> CompiledModel? {
                try tailTier.map { try CompiledModel(packageURL: FWStore(tier: $0).root.appendingPathComponent("block00_int8.mlpackage"), computeUnits: .cpuAndGPU,
                                                     cacheDir: caches.appendingPathComponent($0, isDirectory: true)) }
            }
            if tailTier != nil { try autoreleasepool { _ = try loadTail() } }      // 先建好编译缓存
            if let t = tailTier { say("abort-probe 末尾校验: \(t)") }
            // ★10-09 --abort-probe-tail-every=N(内审 v57「校验本身会不会提高中止率」): 校验包照常载入编好, 只在 i%N==0 的调用上真跑;
            //   其余调用不跑校验, 交替成 A/B(同一轮、同发热状态)。日志里有 tailchk= 的是「跑了」组。默认 1 = 每次都跑(旧口径不变)。
            let tailEvery = max(1, Int(CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-tail-every=") }).map { String($0.dropFirst("--abort-probe-tail-every=".count)) } ?? "1") ?? 1)
            // ★--abort-probe-tail2=<档>:<行>(10-09, ICCE 评审「同开销下末尾 vs 分散抽查」): 同一次调用里再跑第二个校验模型
            //   (export_split_block.py FW_SPLIT_AT=rows), 它只出指定查询行的块输出 [1,K,3072], 与整块输出的这些行比对。
            //   行 = "strideN"(p_k = floor((k+0.5)·T/N), 与 paper_icce/silent/analysis/check_positions.py 同式) 或逗号串; 顺序须与导出时 FW_ROWS 相同。
            //   内存同 tail: 整块与第一个校验模型都放掉之后才载它, 用完即放。行尾记 tail2chk=<1−cos>/<最大差> t2dt= t2load= t2ab=。
            let tail2Arg = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-tail2=") }).map { String($0.dropFirst("--abort-probe-tail2=".count)) }
            var tail2Tier: String? = nil, tail2Rows: [Int] = []
            if let a2 = tail2Arg {
                let parts = a2.split(separator: ":", maxSplits: 1).map(String.init)
                tail2Tier = parts[0]
                let spec = parts.count > 1 ? parts[1] : "stride64"
                if spec.hasPrefix("stride") {
                    let n = Int(spec.dropFirst("stride".count)) ?? 64
                    tail2Rows = (0..<n).map { Int((Double($0) + 0.5) * Double(T) / Double(n)) }
                } else {
                    tail2Rows = spec.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
                }
                if tail2Rows.isEmpty || tail2Rows.contains(where: { $0 < 0 || $0 >= T }) { say("✗ abort-probe 失败: --abort-probe-tail2 行不合法 \(spec)"); return }
            }
            func loadTail2() throws -> CompiledModel? {
                try tail2Tier.map { try CompiledModel(packageURL: FWStore(tier: $0).root.appendingPathComponent("block00_int8.mlpackage"), computeUnits: .cpuAndGPU,
                                                      cacheDir: caches.appendingPathComponent($0, isDirectory: true)) }
            }
            if tail2Tier != nil { try autoreleasepool { _ = try loadTail2() } }    // 先建好编译缓存
            if let t = tail2Tier { say("abort-probe 第二校验: \(t) 行 K=\(tail2Rows.count) \(tail2Rows.prefix(3).map(String.init).joined(separator: ","))…\(tail2Rows.last ?? -1)") }
            let names = Set(m?.model.modelDescription.inputDescriptionsByName.keys ?? [String: MLFeatureDescription]().keys)
            let feed = ["a": a, "b": ctx, "c": mod, "d": rope].filter { names.contains($0.key) }
            let provider = try MLDictionaryFeatureProvider(dictionary: feed.mapValues { MLFeatureValue(multiArray: $0) })
            // ★(10-09 挪到恢复对比分支之前: 原来放在后面, 带 --abort-probe-recover 时根本没跑到) 10-08 晚 --abort-probe-gpucontend(目标「让 iPhone 18 出错」): 整轮在后台线程不停往 GPU 提交小 Metal 核, 跟 Core ML 抢 GPU,
            //   把每段命令缓冲拉长(A14 上被掐都出在 GPU 被长时间占住、挡屏幕合成时)。独立 GPUSpeedo, 轮结束即停。
            let contend = CommandLine.arguments.contains("--abort-probe-gpucontend")
            final class StopFlag: @unchecked Sendable { private let l = NSLock(); private var v = false
                func set() { l.lock(); v = true; l.unlock() }; var isSet: Bool { l.lock(); defer { l.unlock() }; return v } }
            let contendFlag = StopFlag()
            if contend, let sp2 = GPUSpeedo() {
                let box = UncheckedBox(sp2)
                Thread.detachNewThread { while !contendFlag.isSet { _ = box.v.measure() } }
                say("abort-probe GPU 抢占线程已开")
            }
            defer { contendFlag.set() }
            // ★10-09 --abort-probe-gpuhang=<循环次数, 默认 2e9>(目标「让 iPhone 18 出错」): 后台另开一条命令队列, 每隔几秒提交一个超长循环的 Metal 核,
            //   逼 GPU 看门狗超时掐掉它; 看 Core ML 同时在 GPU 上的任务会不会被「连带丢弃」而 predict 照常返回(A14 日志里的「连带丢弃 20」)。
            //   每个卡核的 GPU 错误码写日志(超时 = 看门狗动手了)。
            let msent: MetalSentinel? = CommandLine.arguments.contains("--abort-probe-metalsent") ? MetalSentinel() : nil
            if let ms = msent, let (q, pso, buf) = makeHangKernel() {
                let box = UncheckedBox((q, pso, buf, ms))
                Thread.detachNewThread {
                    var nn: UInt32 = 2000
                    while !contendFlag.isSet {
                        guard let cb = box.v.0.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { break }
                        enc.setComputePipelineState(box.v.1); enc.setBuffer(box.v.2, offset: 0, index: 0); enc.setBytes(&nn, length: 4, index: 1)
                        enc.dispatchThreads(MTLSize(width: 4096, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                        box.v.3.add(cb.error.map { ($0 as NSError).code } ?? 0)
                        Thread.sleep(forTimeInterval: 0.2)
                    }
                }
                say("abort-probe Metal 哨兵线程已开(每 0.2 s 一个极短小核)")
            }
            if let hv = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-gpuhang") }) {
                let loops = UInt32(hv.split(separator: "=").last.flatMap { Double($0) } ?? 2e9)
                if let (q, pso, buf) = makeHangKernel() {
                    let box = UncheckedBox((q, pso, buf))
                    let sayBox = UncheckedBox(say)
                    Thread.detachNewThread {
                        var nn = loops; var k = 0
                        while !contendFlag.isSet {
                            guard let cb = box.v.0.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { break }
                            enc.setComputePipelineState(box.v.1); enc.setBuffer(box.v.2, offset: 0, index: 0); enc.setBytes(&nn, length: 4, index: 1)
                            enc.dispatchThreads(MTLSize(width: 4096, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                            enc.endEncoding(); let t0 = Date(); cb.commit(); cb.waitUntilCompleted(); k += 1
                            let err = cb.error.map { "\(($0 as NSError).code) \($0.localizedDescription)" } ?? "ok"
                            sayBox.v("abort-probe 卡核 #\(k) 用时 \(String(format: "%.1f", -t0.timeIntervalSinceNow))s 结果 \(err)")
                            Thread.sleep(forTimeInterval: 3)
                        }
                    }
                    say("abort-probe GPU 卡核线程已开 循环 \(loops)")
                }
            }
            if !recover.isEmpty && !stages.isEmpty { say("✗ abort-probe 失败: 恢复对比暂不支持分段档"); return }
            say("abort-probe 开始 档=\(tier) iters=\(iters) T=\(T) 输入=\(dumpSub) 发热 \(th()) ui压力=\(CommandLine.arguments.contains("--abort-probe-ui")) 恢复=\(recover.joined(separator: ",")) 单元=\(unitsArg)")
            // ★恢复对比(10-05 ICCE 审稿意见「检出之后能不能恢复」): 每回合 = 产品里的一块。算一次 → 日志(两类都认)判被掐 →
            //   按策略重算, 直到日志干净或用满 8 次; 记每次尝试的对错(对参考)与回合总耗时。策略按回合轮流, 发热漂移各组均摊。
            //   imm = 立即重算; wait = 按产品 retryWaits(10,30,60,…s) 等后重算; reload = 重建 MLModel 后立即重算。
            if !recover.isEmpty {
                let waits = FWPipeline.retryWaits
                for ep in 0..<iters {
                    var bat: Float = -1
                    #if canImport(UIKit)
                    bat = await MainActor.run { UIDevice.current.isBatteryMonitoringEnabled = true; return UIDevice.current.batteryLevel }
                    #endif
                    if bat >= 0 && bat < minBat { say(String(format: "abort-probe 电量 %.0f%% < %.0f%%，停", bat * 100, minBat * 100)); break }
                    let pol = recover[ep % recover.count]
                    let e0 = Date()
                    var seq: [String] = [], finalOK = false, finalFlag = true, waited = 0.0, reloadS = 0.0
                    for att in 0...waits.count {
                        var flagged = false, ok = false, threw = "-", dt = 0.0
                        autoreleasepool {
                            let t0 = Date()
                            var outArr: MLMultiArray?
                            do {
                                let out = try m.model.prediction(from: provider)
                                if let n = out.featureNames.first { outArr = out.featureValue(for: n)?.multiArrayValue }
                            } catch { threw = "threw" }
                            dt = -t0.timeIntervalSinceNow
                            Thread.sleep(forTimeInterval: 0.5)
                            flagged = threw != "-" || FWPipeline.gpuAbortSummary(since: t0) != nil
                            if let o = outArr, let d = try? FWPipeline.detach(o) { ok = 1 - cosMax(d, ref).0 < 1e-3 }
                        }
                        seq.append(String(format: "%@%@%@%.1f", ok ? "ok" : "BAD", flagged ? "/flag" : "", threw == "-" ? "" : "/threw", dt))
                        finalOK = ok; finalFlag = flagged
                        if !flagged || att == waits.count { break }
                        switch pol {
                        case "wait": Thread.sleep(forTimeInterval: waits[att]); waited += waits[att]
                        case "reload":
                            let r0 = Date()
                            m = try loadModel()
                            reloadS += -r0.timeIntervalSinceNow
                        default: break
                        }
                    }
                    say(String(format: "abort-recover ep=%d pol=%@ th=%d bat=%.0f attempts=%d final_ok=%d final_flag=%d total=%.1fs waited=%.0fs reload=%.1fs seq=%@",
                               ep, pol, th(), bat * 100, seq.count, finalOK ? 1 : 0, finalFlag ? 1 : 0, -e0.timeIntervalSinceNow, waited, reloadS, seq.joined(separator: ",")))
                }
                say("✅ abort-probe 完成 发热 \(th())")
                return
            }
            var first: MLMultiArray?
            var windows: [(Int, Date, Date)] = []          // 每次调用的起止，跑完后按时间戳回填日志（查「晚到」的记录）
            var dumped = 0
            let dumpDir = docs.appendingPathComponent("abort_dump", isDirectory: true)
            try? FileManager.default.createDirectory(at: dumpDir, withIntermediateDirectories: true)
            // ★--abort-probe-gpuballast=MB(10-07 机制实验 H): 整轮占着这么多 GPU 内存(MTLBuffer 写满不放),
            //   ★10-07: 放在模型加载之后(放在加载之前 384/768 MB 都在载模型时 App 消失, 无 Jetsam 记录)。检验「GPU 内存压力 ⇒ 掐断变成静默错」—— 重启后热机几乎不错, 占内存后错误是否回来
            let ballastMB = CommandLine.arguments.first(where: { $0.hasPrefix("--abort-probe-gpuballast=") }).flatMap { Int($0.dropFirst("--abort-probe-gpuballast=".count)) } ?? 0
            var ballast: [MTLBuffer] = []
            if ballastMB > 0, let dev = MTLCreateSystemDefaultDevice() {
                for _ in 0..<(ballastMB / 64) {
                    if (os_proc_available_memory() >> 20) < 450 { break }      // 留够调用本身要的余量, 不够就少占(实际占了多少下面会记)
                    guard let b = dev.makeBuffer(length: 64 << 20, options: .storageModeShared) else { break }
                    memset(b.contents(), 0x5a, b.length); ballast.append(b)
                }
                say("abort-probe GPU 压舱 \(ballast.count * 64) MB(要求 \(ballastMB)) wired=\(hostWiredMB()) hostfree=\(hostFreeMB()) availmb=\(Int(os_proc_available_memory() >> 20))")
            }
            defer { ballast.removeAll() }
            let refTok = tokenNorms(ref, T: T, dim: dim)
            var sentBuf: MLMultiArray? = nil         // 哨兵输出缓冲, 整轮复用
            let runStart = Date()
            for i in 0..<iters {
                // ★10-08 晚: iOS 27 上第 1 次调用前就停住(GPU/CPU 都冷着) ⇒ 第 0 次每一步前后落一行, 看停在哪
                if i == 0 { say("abort-probe 步: 读电量前") }
                // 电量（10-05 第 2 轮不调暗 + 界面刷新把 phoneA 耗到没电关机）：每次记下，<30% 自动停
                var bat: Float = -1, chg = "?"
                #if canImport(UIKit)
                (bat, chg) = await MainActor.run { () -> (Float, String) in
                    UIDevice.current.isBatteryMonitoringEnabled = true
                    let st: String
                    switch UIDevice.current.batteryState { case .charging: st = "chg"; case .full: st = "full"; case .unplugged: st = "unp"; default: st = "?" }
                    return (UIDevice.current.batteryLevel, st)
                }
                #endif
                if i == 0 { say("abort-probe 步: 读电量后") }
                let lpm = ProcessInfo.processInfo.isLowPowerModeEnabled ? 1 : 0
                if bat >= 0 && bat < minBat { say(String(format: "abort-probe 电量 %.0f%% < %.0f%%，停", bat * 100, minBat * 100)); break }
                // ★--abort-probe-thermalgate=秒: 发热门控(10-06 预防实验 F)。三台手机上 113 次非「严重」发热的调用 0 错、错全在「严重」⇒
                //   每次调用前若 thermalState ≥ serious, 每 10 s 查一次, 等到降下来或等满上限再算; 记等了多久(门控的代价)。
                var gateWait = 0.0
                if thermalGate > 0 {
                    let g0 = Date()
                    while th() >= 2 && -g0.timeIntervalSinceNow < thermalGate { Thread.sleep(forTimeInterval: 10) }
                    gateWait = -g0.timeIntervalSinceNow
                }
                // ★10-09 --abort-probe-cpuheat=<上限秒>(热是不是触发条件, 与「GPU 刚连续干活」分开): 偶数次调用前只用 CPU 空转把手机烤到 serious(GPU 不碰),
                //   奇数次调用前空闲等降到 fair 及以下; 两组交替, 同机同块。到 critical 立刻停烤。日志另起一行「cpuheat i= arm=H/C」。
                if cpuHeatMax > 0 {
                    let h0 = Date(), hot = i % 2 == 0
                    if hot {
                        let stop = StopFlag()
                        for _ in 0..<ProcessInfo.processInfo.activeProcessorCount {
                            Thread.detachNewThread { var a = 1.0001; while !stop.isSet { for _ in 0..<200_000 { a = a * 1.0000001 + 1e-9 } }; if a < 0 { print(a) } }
                        }
                        while th() < 2 && -h0.timeIntervalSinceNow < cpuHeatMax { Thread.sleep(forTimeInterval: 5) }
                        stop.set()
                    } else {
                        while th() > 1 && -h0.timeIntervalSinceNow < cpuHeatMax { Thread.sleep(forTimeInterval: 10) }
                    }
                    say(String(format: "abort-probe cpuheat i=%d arm=%@ waited=%.0fs th=%d", i, hot ? "H" : "C", -h0.timeIntervalSinceNow, th()))
                }
                let memW = hostWiredMB(), memF = hostFreeMB(), memA = FWPipeline.availableMemoryMB()   // 调用前的全机内存状态(检验「静默错与内存压力有关」)
                let th0 = th()                            // 调用开始前的发热状态(th= 是调用结束时读的; 评审会问「先热后错」还是「先错后热」)
                if i == 0 { say("abort-probe 步: GPU 测速前") }
                let spd = noSpeedo ? -1 : (speedo?.measure() ?? -1)        // 每次调用前测一次 GPU 速度(--abort-probe-nospeedo 跳过)
                if i == 0 { say("abort-probe 步: GPU 测速后 \(spd)") }
                segSpeed = []
                autoreleasepool {
                    if i == 0 { say("abort-probe 步: 进 predict") }
                    let since = Date(), t0 = Date()
                    var threw = "-"
                    var sentInfo = ""
                    var outArr: MLMultiArray?
                    var segDts: [Double] = []
                    do {
                        if stages.isEmpty {
                            if m == nil { m = try loadModel() }           // 末尾校验模式下每次调用前重新加载整块(上一轮校验前放掉了)
                            if sentinel, let (on, od) = m.model.modelDescription.outputDescriptionsByName.first,
                               let cs = od.multiArrayConstraint {
                                // ★10-08: 整轮只分配一次(每次新建 + 逐元素 NSNumber 读回 ⇒ 第 2 次调用时 App 无声消失, 疑被内存杀)
                                // --abort-probe-sentinel-fresh(10-08 晚): 复用同一块缓冲时第 2 次调用永不返回(phoneA 两轮复现, 44 分钟) ⇒ 每次新建、读完即放
                                if sentFresh { sentBuf = nil }      // 先放旧的再建新的, 免得两块 140 MB 同时在
                                if sentBuf == nil { sentBuf = try MLMultiArray(shape: cs.shape, dataType: cs.dataType) }
                                let bk = sentBuf!
                                let isH = cs.dataType == .float16
                                if isH { let q = bk.dataPointer.bindMemory(to: UInt16.self, capacity: bk.count); for j in 0..<bk.count { q[j] = 0x7E00 } }
                                else { let q = bk.dataPointer.bindMemory(to: Float.self, capacity: bk.count); for j in 0..<bk.count { q[j] = .nan } }
                                let opt = MLPredictionOptions(); opt.outputBackings = [on: bk]
                                let out = try m.model.prediction(from: provider, options: opt)
                                outArr = out.featureValue(for: on)?.multiArrayValue
                                var sentTok = 0, sentFirst = -1
                                if let o = outArr {
                                    let same = (o.dataPointer == bk.dataPointer)
                                    let row = o.count / max(T, 1)
                                    for t in 0..<T {
                                        var allNaN = true
                                        for k in stride(from: 0, to: row, by: max(row / 8, 1)) {
                                            let idx = t * row + k
                                            let isNaN: Bool = isH ? ((o.dataPointer.bindMemory(to: UInt16.self, capacity: o.count)[idx] & 0x7C00) == 0x7C00 && (o.dataPointer.bindMemory(to: UInt16.self, capacity: o.count)[idx] & 0x03FF) != 0)
                                                                  : o.dataPointer.bindMemory(to: Float.self, capacity: o.count)[idx].isNaN
                                            if !isNaN { allNaN = false; break }
                                        }
                                        if allNaN { sentTok += 1; if sentFirst < 0 { sentFirst = t } }
                                    }
                                    sentInfo = "sent=\(sentTok)/\(T) sfirst=\(sentFirst) samebuf=\(same ? 1 : 0) dtype=\(cs.dataType.rawValue) fresh=\(sentFresh ? 1 : 0)"
                                }
                            } else {
                                let out = try m.model.prediction(from: provider)
                                if let n = out.featureNames.first { outArr = out.featureValue(for: n)?.multiArrayValue }
                            }
                        } else { (outArr, segDts) = try runChain(i) }
                    } catch { threw = String(describing: error).replacingOccurrences(of: "\n", with: " ").prefix(160).description }
                    let dt = -t0.timeIntervalSinceNow
                    if !segDts.isEmpty { threw += " seg=" + segDts.map { String(format: "%.1f", $0) }.joined(separator: "/") + " load=" + loadDts.map { String(format: "%.1f", $0) }.joined(separator: "/") }
                    let tEnd = Date()
                    windows.append((i, since, tEnd))
                    Thread.sleep(forTimeInterval: 0.5)            // 产品代码等 0.2 s 让 Metal 回调的错误日志落盘；这里等 0.5 s
                    let abort = FWPipeline.gpuAbortSummary(since: since) ?? "-"
                    var stat = "无输出"
                    if let o = outArr, let d = try? FWPipeline.detach(o) {
                        let b = d.fp16Buffer
                        var zeros = 0, bad = 0, ss = 0.0
                        for j in 0..<b.count { let v = Double(b[j]); if v == 0 { zeros += 1 }; if !v.isFinite { bad += 1 } else { ss += v * v } }
                        let (c, mx) = cosMax(d, ref)
                        var c0 = 1.0
                        if let f = first { c0 = cosMax(d, f).0 } else { first = d }
                        var amax = 0.0
                        for j in 0..<b.count { let v = abs(Double(b[j])); if v.isFinite && v > amax { amax = v } }
                        // 错在哪些 token：每个 token 的 L2 范数与参考差 >1% 算坏（错误结构，用来设计按 token 分块的检测）
                        let tok = tokenNorms(d, T: T, dim: dim)
                        var nbad = 0, firstBad = -1, lastBad = -1, runs = 0, prevBad = false
                        for t in 0..<T {
                            let isBad = abs(tok[t] - refTok[t]) > 0.01 * max(refTok[t], 1e-6)
                            if isBad { nbad += 1; if firstBad < 0 { firstBad = t }; lastBad = t; if !prevBad { runs += 1 } }
                            prevBad = isBad
                        }
                        let freeGB = Double((try? docs.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage ?? 0) / 1e9
                        if 1 - c > 1e-3 && dumped < 8 && freeGB > 1.0 {   // 只存错的，最多 8 份（每份约 70MB）；剩余 <1GB 不存（10-05 phoneB 被存满）
                            let url = dumpDir.appendingPathComponent("out_\(i).f16")
                            Data(buffer: UnsafeBufferPointer(start: b.baseAddress, count: b.count)).withUnsafeBytes { try? Data($0).write(to: url) }
                            dumped += 1
                        }
                        stat = String(format: "zero=%.4f naninf=%d rms=%.4f amax=%.3g ref_1mcos=%.3e ref_maxdiff=%.4g first_1mcos=%.3e badtok=%d/%d first=%d last=%d runs=%d", Double(zeros) / Double(max(b.count, 1)), bad, (ss / Double(max(b.count, 1))).squareRoot(), amax, 1 - c, mx, 1 - c0, nbad, T, firstBad, lastBad, runs)
                        if tailTier != nil && i % tailEvery == 0 {
                            m = nil                                       // 放掉整块模型(连同它的中间缓冲), 再载校验模型
                            let tl0 = Date()
                            let tm = try? loadTail() ?? nil
                            let tload = -tl0.timeIntervalSinceNow
                            let tt0 = Date()
                            var tailS = String(format: "tailchk=threw tload=%.2f", tload)
                            if let tm, let to = try? tm.model.prediction(from: provider), let n = to.featureNames.first,
                               let tv = to.featureValue(for: n)?.multiArrayValue, let td = try? FWPipeline.detach(tv) {
                                let tb = td.fp16Buffer, K = tb.count / dim, off = (T - K) * dim
                                var dot = 0.0, na = 0.0, nb = 0.0, mxd = 0.0
                                for j in 0..<tb.count {
                                    let x = Double(b[off + j]), y = Double(tb[j])
                                    if x.isFinite && y.isFinite { dot += x * y; na += x * x; nb += y * y; mxd = max(mxd, abs(x - y)) } else { mxd = .infinity }
                                }
                                let tdt = -tt0.timeIntervalSinceNow
                                Thread.sleep(forTimeInterval: 0.3)
                                let tab = FWPipeline.gpuAbortSummary(since: tt0) == nil ? 0 : 1
                                tailS = String(format: "tailchk=%.3e/%.4g tdt=%.2f tload=%.2f tab=%d", 1 - dot / max((na * nb).squareRoot(), 1e-30), mxd, tdt, tload, tab)
                            }
                            stat += " " + tailS
                        }
                        if tail2Tier != nil && i % tailEvery == 0 {
                            m = nil                                       // (只给 tail2 时也要先放掉整块; 第一个校验模型在上面的块里已出作用域)
                            let tl0 = Date()
                            var t2S = "tail2chk=threw"
                            autoreleasepool {
                                let tm2 = try? loadTail2() ?? nil
                                let tload2 = -tl0.timeIntervalSinceNow
                                t2S = String(format: "tail2chk=threw t2load=%.2f", tload2)
                                let tt0 = Date()
                                if let tm2, let to = try? tm2.model.prediction(from: provider), let n = to.featureNames.first,
                                   let tv = to.featureValue(for: n)?.multiArrayValue, let td = try? FWPipeline.detach(tv) {
                                    let tdt2 = -tt0.timeIntervalSinceNow
                                    let tb = td.fp16Buffer, K = tb.count / dim
                                    if K != tail2Rows.count {
                                        t2S = String(format: "tail2chk=rowsmismatch K=%d want=%d t2load=%.2f", K, tail2Rows.count, tload2)
                                    } else {
                                        var dot = 0.0, na = 0.0, nb = 0.0, mxd = 0.0
                                        for (r, row) in tail2Rows.enumerated() {
                                            let off = row * dim, o2 = r * dim
                                            for j in 0..<dim {
                                                let x = Double(b[off + j]), y = Double(tb[o2 + j])
                                                if x.isFinite && y.isFinite { dot += x * y; na += x * x; nb += y * y; mxd = max(mxd, abs(x - y)) } else { mxd = .infinity }
                                            }
                                        }
                                        Thread.sleep(forTimeInterval: 0.3)
                                        let tab2 = FWPipeline.gpuAbortSummary(since: tt0) == nil ? 0 : 1
                                        t2S = String(format: "tail2chk=%.3e/%.4g t2dt=%.2f t2load=%.2f t2ab=%d", 1 - dot / max((na * nb).squareRoot(), 1e-30), mxd, tdt2, tload2, tab2)
                                    }
                                }
                            }
                            stat += " " + t2S
                        }
                    }
                    let spdS = String(format: "%.2f", spd) + (segSpeed.isEmpty ? "" : "/" + segSpeed.map { String(format: "%.2f", $0) }.joined(separator: "/"))
                    if !sentInfo.isEmpty { stat += " " + sentInfo }
                    if let ms = msent { stat += " " + ms.window(since, tEnd.addingTimeInterval(0.5)) }
                    say(String(format: "abort-probe i=%d th=%d th0=%d bat=%.0f pwr=%@ lpm=%d wired=%d hostfree=%d availmb=%d uptime=%.0f gpuspd=%@ gate=%.0f dt=%.2fs threw=%@ oslog=%@ %@", i, th(), th0, bat * 100, chg, lpm, memW, memF, memA, ProcessInfo.processInfo.systemUptime, spdS, gateWait, dt, threw, abort, stat))
                }
            }
            // 回填：整轮的 GPU 错误日志一次取回，按时间戳分到各次调用 [起, 止+0.5s]；与当时读到的对比 ⇒ 晚到 / 真没记
            let logs = gpuLogsWithDates(since: runStart)
            for (i, a, b) in windows {
                let inW = logs.filter { $0.0 >= a && $0.0 <= b.addingTimeInterval(0.5) }
                let ii = inW.filter { $0.1.contains("ImpactingInteractivity") }.count, iv = inW.filter { $0.1.contains("InnocentVictim") }.count
                if ii + iv > 0 { say("abort-probe post i=\(i) ii=\(ii) iv=\(iv) first_dt=\(String(format: "%.2f", inW.first!.0.timeIntervalSince(a)))s dur=\(String(format: "%.2f", b.timeIntervalSince(a)))s") }
            }
            for (i, k, a0, b0) in stageWins {
                let inW = logs.filter { $0.0 >= a0 && $0.0 <= b0.addingTimeInterval(0.5) }
                let ii = inW.filter { $0.1.contains("ImpactingInteractivity") }.count, iv = inW.filter { $0.1.contains("InnocentVictim") }.count
                if ii + iv > 0 { say("abort-probe postseg i=\(i) seg=\(k) ii=\(ii) iv=\(iv) dur=\(String(format: "%.2f", b0.timeIntervalSince(a0)))s") }
            }
            say("abort-probe post 完成 共 \(logs.count) 条 GPU 错误日志")
            say("✅ abort-probe 完成 发热 \(th())")
        } catch { say("✗ abort-probe 失败: \(error)") }
    }


    static func cosMax(_ x: MLMultiArray, _ y: MLMultiArray) -> (Double, Double) {
        let a = x.fp16Buffer, b = y.fp16Buffer
        var dot = 0.0, na = 0.0, nb = 0.0, mx = 0.0
        for j in 0..<min(a.count, b.count) {
            let p = Double(a[j]), q = Double(b[j])
            dot += p * q; na += p * p; nb += q * q
            let dd = abs(p - q); if dd > mx || dd.isNaN { mx = dd }
        }
        return (dot / (na.squareRoot() * nb.squareRoot() + 1e-12), mx)
    }

    /// 本进程 phys_footprint(MB)。⚠️不含 mmap 权重; GPU/IOSurface 是否计入以实测为准, 与 os_proc_available_memory 一起看。
    static func footprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint / 1_048_576) : -1
    }

    /// 25ms 采一次内存(最小可用 / 最大 footprint / 最小全机可腾)。
    final class MemSampler: @unchecked Sendable {
        private let q = DispatchQueue(label: "blockprobe.mem")
        private var timer: DispatchSourceTimer?
        private var minAvail = Int.max, maxFoot = 0, minHost = Int.max
        private func sample() {
            minAvail = min(minAvail, FWPipeline.availableMemoryMB())
            maxFoot = max(maxFoot, SegTest.footprintMB())
            minHost = min(minHost, SegTest.hostFreeMB())
        }
        func start() {
            let tm = DispatchSource.makeTimerSource(queue: q)
            tm.schedule(deadline: .now(), repeating: .milliseconds(25))
            tm.setEventHandler { [weak self] in self?.sample() }
            timer = tm; tm.resume()
        }
        func stop() -> (minAvail: Int, maxFoot: Int, minHost: Int) {
            timer?.cancel(); timer = nil
            return q.sync { sample(); return (minAvail, maxFoot, minHost) }
        }
    }

    /// MLComputePlan: 每个算子被派到哪个设备 + CoreML 自己估的相对耗时权重(**估计值, 不是实测**)。
    ///   按算子类型汇总; linear / scaled_dot_product_attention / gelu 逐个按图中顺序列出(自注意力 q,k,v,o → 交叉 q,k,v,o → FFN 两个)。

    /// 全机能腾出的物理内存(free+inactive+purgeable), MB。ANE 程序的内存不算进本进程配额, 只能看全机。
    /// 全机锁定(wired)内存 MB(GPU 占的内存属于这类; 10-07 phoneA 静默错 93–99% 那段被杀前 wired 2.57 GB、重启后热机 0 错)
    static func hostWiredMB() -> Int {
        var st = vm_statistics64(); var cnt = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let r = withUnsafeMutablePointer(to: &st) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(cnt)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &cnt) } }
        guard r == KERN_SUCCESS else { return -1 }
        return Int(UInt64(st.wire_count) * UInt64(vm_kernel_page_size) / 1_048_576)
    }

    static func hostFreeMB() -> Int {
        var st = vm_statistics64(); var cnt = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let r = withUnsafeMutablePointer(to: &st) { $0.withMemoryRebound(to: integer_t.self, capacity: Int(cnt)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &cnt) } }
        guard r == KERN_SUCCESS else { return -1 }
        let pages = UInt64(st.free_count) + UInt64(st.inactive_count) + UInt64(st.purgeable_count)
        return Int(pages * UInt64(vm_kernel_page_size) / 1_048_576)
    }

}
