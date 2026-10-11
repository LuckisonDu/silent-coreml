// ICCE 实验 V2(安卓「单个 GPU 任务超过驱动超时」): 一个 BATCH_MATMUL 模型 X(1,1,M,K)·Y(1,1,K,N) → (1,1,M,N),
// 两个输入都是运行时张量, 由本探针按确定性整数哈希现场生成(不用推大文件), 参考输出由 服务器上 numpy 按同一公式算好(ref.f32)。
// 目的: 让「一次 Invoke 里的单个 GPU kernel」本身就跑 8~20 s(超过 Mali kbase 计算 job 约 5 s 的硬停), 看 LiteRT 返回什么:
//   报错(Invoke 非 0 / error reporter 有消息) · 静默错(返回成功但输出不对) · 正确(驱动没掐) · 崩溃(进程没了 / 整机重启)。
// 输入值 = ((hash(i, seed) % 17) - 8) / 8, 全是 1/8 的整数倍 ⇒ fp16 精确, 乘积与部分和在 fp32 里也精确(|和|·64 < 2^24)。
// 防「旧的正确值」盖住错误: --alt 1 时奇数次 X×0.5(参考也×0.5, 精确); 每次调用前输出缓冲填哨兵。
// 每行: i dt threw msgs sentinel naninf zero ref_1mcos ref_maxdiff badrow/M first last same_prev + 温控/电量/内存。
// 另有 CONV_2D 版(大卷积核, 权重是模型常量, 只有一个输入): 计算/字节比高, 单个 kernel 很长而内存不大。
// 用法: icce_longjob_probe --model mm.tflite --ref ref.f32 --n 20 [--prec mixed|fp32|fp16]
//        [--backend gpu|cpu] [--alt 1] [--gap 0] [--mem-guard 350] [--out log.txt] [--cache-dir d] [--dump-bad 1 --dump-dir d]
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <functional>

#include "tflite/core/c/c_api.h"
#include "tflite/delegates/gpu/delegate.h"

static FILE* g_log = nullptr;
static std::atomic<int> g_err_count{0};
static std::string now_str() {
  struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); struct tm tm; gmtime_r(&ts.tv_sec, &tm);
  char b[64]; strftime(b, sizeof b, "%Y-%m-%d %H:%M:%S", &tm);
  char c[80]; snprintf(c, sizeof c, "%s.%03ld +0000", b, ts.tv_nsec / 1000000); return c;
}
static void logf(const char* fmt, ...) {
  char buf[8192]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
  std::string line = "[" + now_str() + "] " + buf;
  fprintf(stdout, "%s\n", line.c_str()); fflush(stdout);
  if (g_log) { fprintf(g_log, "%s\n", line.c_str()); fflush(g_log); fsync(fileno(g_log)); }
}
static void reporter(void*, const char* fmt, va_list ap) {
  char buf[4096]; vsnprintf(buf, sizeof buf, fmt, ap); g_err_count++;
  for (char* p = buf; *p; ++p) if (*p == '\n') *p = ' ';
  logf("tflite-msg %s", buf);
}
static double mono() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static std::string run_cmd(const char* cmd) {
  FILE* p = popen(cmd, "r"); if (!p) return ""; std::string o; char b[1024];
  while (fgets(b, sizeof b, p)) o += b; pclose(p); return o;
}
static std::string field_after(const std::string& s, const std::string& key) {
  size_t i = s.find(key); if (i == std::string::npos) return "";
  i += key.size(); while (i < s.size() && (s[i] == ' ' || s[i] == ':' || s[i] == '=')) i++;
  size_t j = i; while (j < s.size() && s[j] != '\n' && s[j] != ',' && s[j] != '}' && s[j] != ' ') j++;
  return s.substr(i, j - i);
}
static std::string power_state() {   // shell 用户读不到 thermal_zone ⇒ 走 dumpsys(同 icce_gpu_probe)
  std::string th = run_cmd("dumpsys thermalservice 2>/dev/null");
  size_t h = th.find("Current temperatures from HAL"); std::string cur = h == std::string::npos ? th : th.substr(h);
  std::string ap = "-"; size_t i = cur.find("mName=AP,");
  if (i != std::string::npos) { size_t v = cur.rfind("mValue=", i); if (v != std::string::npos) ap = field_after(cur.substr(v), "mValue"); }
  std::string bt = run_cmd("dumpsys battery 2>/dev/null");
  char b[200]; snprintf(b, sizeof b, "tstat=%s AP=%s bat=%s", field_after(th, "Thermal Status").c_str(), ap.c_str(), field_after(bt, "  level").c_str());
  return b;
}
static long mem_avail_mb() {
  FILE* f = fopen("/proc/meminfo", "r"); if (!f) return -1; char l[256]; long v = -1;
  while (fgets(l, sizeof l, f)) if (!strncmp(l, "MemAvailable:", 13)) { v = atol(l + 13) / 1024; break; }
  fclose(f); return v;
}
static std::atomic<long> g_min_mem{1L << 40};
static void mem_guard(long thr_mb) {   // 10-08 A12 OOM 时三星内核 panic 重启 ⇒ 可用内存低于门槛就自杀
  for (;;) {
    long m = mem_avail_mb();
    if (m > 0 && m < g_min_mem) g_min_mem = m;
    if (m > 0 && m < thr_mb) { logf("mem-guard 触发: MemAvailable %ldMB < %ldMB, 探针自杀保护手机", m, thr_mb); _exit(42); }
    usleep(100000);
  }
}
// 与 服务器上 build_longjob_tflite.py 的 numpy 版逐位相同: lowbias32(i ^ seed·0x9E3779B9)
static inline uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
static void gen(std::vector<float>& v, size_t n, uint32_t seed) {
  v.resize(n); uint32_t s = seed * 0x9E3779B9U;
  for (size_t i = 0; i < n; ++i) v[i] = ((int)(hash32((uint32_t)i ^ s) % 17u) - 8) / 8.0f;
}

int main(int argc, char** argv) {
  std::string model, ref, out = "longjob.txt", backend = "gpu", prec = "mixed", cache_dir, dump_dir = ".";
  long M = 0, K = 0, N = 0; int n_calls = 20, alt = 1, dump_bad = 1, threads = 4; double gap = 0; long guard_mb = 350;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--model") model = v; else if (k == "--ref") ref = v; else if (k == "--out") out = v;
    else if (k == "--M") M = atol(v.c_str()); else if (k == "--K") K = atol(v.c_str()); else if (k == "--N") N = atol(v.c_str());
    else if (k == "--n") n_calls = atoi(v.c_str()); else if (k == "--alt") alt = atoi(v.c_str());
    else if (k == "--backend") backend = v; else if (k == "--prec") prec = v; else if (k == "--gap") gap = atof(v.c_str());
    else if (k == "--mem-guard") guard_mb = atol(v.c_str()); else if (k == "--cache-dir") cache_dir = v;
    else if (k == "--dump-bad") dump_bad = atoi(v.c_str()); else if (k == "--dump-dir") dump_dir = v;
    else if (k == "--threads") threads = atoi(v.c_str());
    else { fprintf(stderr, "未知参数 %s\n", k.c_str()); return 2; }
  }
  if (model.empty()) { fprintf(stderr, "要 --model\n"); return 2; }
  g_log = fopen(out.c_str(), "a");
  std::thread(mem_guard, guard_mb).detach();
  logf("longjob 开始 model=%s n=%d backend=%s prec=%s alt=%d %s mem=%ldMB", model.c_str(), n_calls, backend.c_str(), prec.c_str(), alt,
       power_state().c_str(), mem_avail_mb());
  std::vector<float> X, Y, R;

  TfLiteModel* m = TfLiteModelCreateFromFile(model.c_str());
  if (!m) { logf("模型载入失败"); return 3; }
  TfLiteInterpreterOptions* opt = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetErrorReporter(opt, reporter, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(opt, threads);
  TfLiteDelegate* del = nullptr; std::string token;
  if (backend == "gpu") {
    TfLiteGpuDelegateOptionsV2 go = TfLiteGpuDelegateOptionsV2Default();
    go.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE; go.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    if (prec == "fp16") { go.is_precision_loss_allowed = 1; go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY; }
    else if (prec == "mixed") {   // F16 存、F32 算
      go.is_precision_loss_allowed = -1; go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
      go.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION; go.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    } else { go.is_precision_loss_allowed = 0; go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION; }
    go.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    go.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
    if (!cache_dir.empty()) {
      go.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
      token = "lj_" + std::to_string(std::hash<std::string>()(model)) + "_" + prec;
      go.serialization_dir = cache_dir.c_str(); go.model_token = token.c_str();
    }
    go.max_delegated_partitions = 1;
    del = TfLiteGpuDelegateV2Create(&go);
    if (!del) { logf("GPU delegate 创建失败"); return 4; }
    TfLiteInterpreterOptionsAddDelegate(opt, del);
  }
  double t0 = mono();
  TfLiteInterpreter* it = TfLiteInterpreterCreate(m, opt);
  if (!it) { logf("解释器创建失败(GPU delegate 没接上?)"); return 5; }
  if (TfLiteInterpreterAllocateTensors(it) != kTfLiteOk) { logf("AllocateTensors 失败"); return 6; }
  logf("建图 %.1fs 最低 mem=%ldMB 现 mem=%ldMB", mono() - t0, (long)g_min_mem, mem_avail_mb());
  // 尺寸全从模型张量读: 输入 0 = X(种子 1), 有输入 1 则 = Y(种子 2, matmul 版; conv 版权重是模型里的常量); 输出最后一维 = 一「行」
  int nin = TfLiteInterpreterGetInputTensorCount(it);
  TfLiteTensor* tx = TfLiteInterpreterGetInputTensor(it, 0);
  const TfLiteTensor* to = TfLiteInterpreterGetOutputTensor(it, 0);
  size_t no = TfLiteTensorByteSize(to) / 4; N = TfLiteTensorDim(to, TfLiteTensorNumDims(to) - 1); M = (long)(no / N);
  gen(X, TfLiteTensorByteSize(tx) / 4, 1);
  if (nin > 1) { TfLiteTensor* ty = TfLiteInterpreterGetInputTensor(it, 1); gen(Y, TfLiteTensorByteSize(ty) / 4, 2);
                 TfLiteTensorCopyFromBuffer(ty, Y.data(), Y.size() * 4); std::vector<float>().swap(Y); }
  bool have_ref = false;
  if (!ref.empty()) {
    FILE* f = fopen(ref.c_str(), "rb");
    if (f) { R.resize(no); have_ref = fread(R.data(), 4, no, f) == no; fclose(f); }
    if (!have_ref) logf("⚠️参考 %s 读不到或大小不对", ref.c_str());
  }
  logf("输入 %d 个, X %zu 个数, 输出 %ld 行 × %ld, mem=%ldMB", nin, X.size(), M, N, mem_avail_mb());
  const float SENT = -12345.678f;
  float* op = (float*)TfLiteTensorData(to);
  uint64_t prev_hash = 0; int nbad = 0, dumped = 0;
  for (int iter = 0; iter < n_calls; ++iter) {
    bool half = alt && (iter & 1);
    float* xin = (float*)TfLiteTensorData(tx);
    for (size_t k = 0; k < X.size(); ++k) xin[k] = half ? X[k] * 0.5f : X[k];
    for (size_t k = 0; k < no; ++k) op[k] = SENT;
    std::string pw0 = power_state(); long mem0 = mem_avail_mb(); int e0 = g_err_count;
    logf("  i=%d 开始 Invoke", iter);
    double s = mono();
    TfLiteStatus st = TfLiteInterpreterInvoke(it);
    double dt = mono() - s;
    int nerr = g_err_count - e0;
    size_t sent = 0, naninf = 0, zero = 0;
    for (size_t k = 0; k < no; ++k) { float v = op[k]; if (v == SENT) sent++; else if (!isfinite(v)) naninf++; else if (v == 0.f) zero++; }
    double xy = 0, xx = 0, yy = 0, md = 0; long badrow = -1, fb = -1, lb = -1; double maxrel = 0;
    if (have_ref) {
      badrow = 0; float sc = half ? 0.5f : 1.f;
      for (long r = 0; r < M; ++r) {
        double num = 0, den = 0;
        for (long c = 0; c < N; ++c) {
          double a = op[r * N + c], b = (double)R[r * N + c] * sc;
          if (a == SENT || !isfinite(a)) a = 0;
          xy += a * b; xx += a * a; yy += b * b; double d = fabs(a - b); if (d > md) md = d;
          num += (a - b) * (a - b); den += b * b;
        }
        double rel = den > 0 ? sqrt(num / den) : (num > 0 ? 1e9 : 0);
        if (op[r * N] == SENT) rel = 1e9;
        if (!(rel <= 0.01)) { badrow++; if (fb < 0) fb = r; lb = r; }
        if (rel > maxrel) maxrel = rel;
      }
    }
    double omc = (xx > 0 && yy > 0) ? 1 - xy / sqrt(xx * yy) : 1.0;
    uint64_t h = 1469598103934665603ull; const uint8_t* bp = (const uint8_t*)op;
    for (size_t k = 0; k < no * 4; ++k) { h ^= bp[k]; h *= 1099511628211ull; }   // 逐字节(只抽样会漏掉 ×0.5 这种只改指数字节的变化)
    bool same_prev = iter > 0 && h == prev_hash; prev_hash = h;
    bool bad = st != kTfLiteOk || nerr || sent || naninf || (have_ref && (badrow > 0 || omc > 1e-4));
    if (bad) nbad++;
    logf("longjob i=%d %s dt=%.3fs threw=%s msgs=%d sentinel=%zu naninf=%zu zero=%.4f ref_1mcos=%.3e ref_maxdiff=%.4g badrow=%ld/%ld first=%ld last=%ld maxrel=%.3g same_prev=%d "
         "pw0=[%s] pw=[%s] mem0=%ld mem=%ld minmem=%ld bad=%d",
         iter, half ? "in=half" : "in=full", dt, st == kTfLiteOk ? "-" : std::to_string((int)st).c_str(), nerr, sent, naninf, (double)zero / no,
         omc, md, badrow, M, fb, lb, maxrel, (int)same_prev, pw0.c_str(), power_state().c_str(), mem0, mem_avail_mb(), (long)g_min_mem, (int)bad);
    if (bad && dumped < dump_bad) {
      std::string p = dump_dir + "/bad_out_i" + std::to_string(iter) + ".f32";
      FILE* f = fopen(p.c_str(), "wb"); if (f) { fwrite(op, 4, no, f); fclose(f); dumped++; logf("存坏输出 %s", p.c_str()); }
    }
    if (gap > 0) usleep((useconds_t)(gap * 1e6));
  }
  logf("longjob 结束 坏 %d 次 %s mem=%ldMB", nbad, power_state().c_str(), mem_avail_mb());
  TfLiteInterpreterDelete(it); if (del) TfLiteGpuDelegateV2Delete(del); TfLiteInterpreterOptionsDelete(opt); TfLiteModelDelete(m);
  if (g_log) fclose(g_log);
  return 0;
}
