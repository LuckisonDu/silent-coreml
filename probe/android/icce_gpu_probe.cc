// ICCE 实验 V(安卓对照)探针: 在 adb shell 里直接跑的原生可执行文件, 不用写 App。
// 同一个 Wan Turbo 第 0 块(.tflite, export_block0_tflite.py 导出)用 LiteRT GPU delegate 反复推理,
// 每次与 torch fp32 参考比对, 逐次记一行日志(格式尽量对齐 iPhone 探针 SegTest.swift --abort-probe 的 abort_probe.txt)。
//
// 每行: i dt threw(Invoke 返回码 + 期间 error reporter 收到几条) zero naninf sentinel(输出里仍是哨兵值的个数 = 没被写回)
//       rms amax ref_1mcos ref_maxdiff first_1mcos(对第 0 次输出) badtok/T first last tail64(逐 token 相对范数差 >1%)
//       + 发热(所有 thermal_zone 的最高值与 GPU 区) 电量 充电状态 MemAvailable GPU 频率/利用率(读得到才记)。
// ⚠️GPU delegate 输出是 GPU 缓冲读回 CPU: 若 GPU job 被驱动硬停, 缓冲里可能是**上一次的旧数据**(Arm 员工说法「stale data」),
//   而同一输入的上一次结果是对的 ⇒ 静默错会被「旧的正确值」盖住。--alt 1 时奇数次把 hidden 输入换成 0.5 倍(参考另算不了,
//   奇数次与 --ref2(输入×0.5 的 torch 参考)比, 并记 same_prev) —— 防「旧的正确值」盖住静默错; 默认 0 = 与 iPhone 一样每次同一输入。
//
// --model 可给分段: "f1:a,c,d>h;f2:h,b>h;f3:h,c>n;…"(输入字母 > 输出字母; a b c d = 输入文件, z = 全 0), 依次调用; --reload 1 每次每段现建现删(省内存)。
// 用法: icce_gpu_probe --model m.tflite --inp <目录含 stem_out.f16 ctx.f16 mod3.f16 rope.f16> --ref ref_out.f16
//         --T 4550 --n 100 [--backend gpu|cpu] [--prec fp32|fp16] [--api cl|gl|auto] [--out log.txt]
//         [--warm-model att.tflite --warm-ap 55 --warm-min 300 --warm-max 600 --keepwarm 2](热机: 预热到 AP≥55°C 或 tstat>0, 调用间空转保温)
//         [--ref2 ref_out_s0.5.f16] [--reload 0|1] [--mem-guard 350] [--min-bat 30] [--gap 0] [--dump-bad 1] [--dump-dir d] [--threads 4] [--alt 0]
#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <functional>
#include <map>

#include "tflite/core/c/c_api.h"
#include "tflite/delegates/gpu/delegate.h"

static const int D = 3072;
static FILE* g_log = nullptr;
static int g_err_count = 0;
static std::string g_last_err;

static std::string now_str() {
  time_t t = time(nullptr); struct tm tm; gmtime_r(&t, &tm);
  char b[64]; strftime(b, sizeof b, "%Y-%m-%d %H:%M:%S +0000", &tm); return b;
}
static void logf(const char* fmt, ...) {
  char buf[8192]; va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
  std::string line = "[" + now_str() + "] " + buf;
  fprintf(stdout, "%s\n", line.c_str()); fflush(stdout);
  if (g_log) { fprintf(g_log, "%s\n", line.c_str()); fflush(g_log); fsync(fileno(g_log)); }
}
static void reporter(void*, const char* fmt, va_list ap) {
  char buf[4096]; vsnprintf(buf, sizeof buf, fmt, ap);
  g_err_count++; g_last_err = buf;
  for (char* p = buf; *p; ++p) if (*p == '\n') *p = ' ';
  logf("tflite-msg %s", buf);
}
static double mono() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static bool read_f16(const std::string& p, size_t n, std::vector<float>& out) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) { logf("读不到 %s", p.c_str()); return false; }
  std::vector<_Float16> h(n); size_t got = fread(h.data(), 2, n, f); fclose(f);
  if (got != n) { logf("%s 大小不对: 读到 %zu 个 fp16, 应为 %zu", p.c_str(), got, n); return false; }
  out.resize(n); for (size_t i = 0; i < n; ++i) out[i] = (float)h[i]; return true;
}
static std::string read_sys(const std::string& p) {
  FILE* f = fopen(p.c_str(), "r"); if (!f) return "";
  char b[512]; size_t n = fread(b, 1, sizeof b - 1, f); fclose(f); b[n] = 0;
  std::string s(b); while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
  for (auto& c : s) if (c == '\n') c = ';';
  return s;
}
// 所有 thermal_zone: 返回 "max=xx.x gpu=yy.y" 并在 all 里给 type:temp 全表
static std::string thermal(std::string* all) {
  double mx = -1e9, gpu = -1e9; std::string full;
  DIR* d = opendir("/sys/class/thermal");
  if (d) {
    struct dirent* e;
    while ((e = readdir(d))) {
      if (strncmp(e->d_name, "thermal_zone", 12)) continue;
      std::string base = std::string("/sys/class/thermal/") + e->d_name;
      std::string t = read_sys(base + "/temp"), ty = read_sys(base + "/type");
      if (t.empty()) continue;
      double v = atof(t.c_str()); if (v > 1000) v /= 1000.0;
      mx = std::max(mx, v);
      std::string tl = ty; std::transform(tl.begin(), tl.end(), tl.begin(), ::tolower);
      if (tl.find("g3d") != std::string::npos || tl.find("gpu") != std::string::npos) gpu = std::max(gpu, v);
      char b[96]; snprintf(b, sizeof b, "%s:%.1f,", ty.c_str(), v); full += b;
    }
    closedir(d);
  }
  if (all) *all = full;
  char b[96];
  snprintf(b, sizeof b, "tmax=%.1f tgpu=%s", mx > -1e8 ? mx : -1.0, gpu > -1e8 ? std::to_string(gpu).substr(0, 5).c_str() : "-");
  return b;
}
// Android 11 上 shell 用户读不到 /sys/class/power_supply/battery/*、thermal_zone 的 type(A12 实测 Permission denied)
// ⇒ 走 dumpsys(shell 用户可用): thermalservice 的「Thermal Status」与 HAL 温度(AP=芯片, BAT=电池), battery 的 level/temperature/status。
static std::string run_cmd(const char* cmd) {
  FILE* p = popen(cmd, "r"); if (!p) return ""; std::string o; char b[1024];
  while (fgets(b, sizeof b, p)) o += b; pclose(p); return o;
}
static std::string field_after(const std::string& s, const std::string& key) {   // "key: value" 或 "key=value,"
  size_t i = s.find(key); if (i == std::string::npos) return "";
  i += key.size(); while (i < s.size() && (s[i] == ' ' || s[i] == ':' || s[i] == '=')) i++;
  size_t j = i; while (j < s.size() && s[j] != '\n' && s[j] != ',' && s[j] != '}' && s[j] != ' ') j++;
  return s.substr(i, j - i);
}
static int g_bat = -1;
static std::string power_state() {
  std::string th = run_cmd("dumpsys thermalservice 2>/dev/null");
  size_t h = th.find("Current temperatures from HAL"); std::string cur = h == std::string::npos ? th : th.substr(h);
  auto temp_of = [&](const char* name) {
    std::string k = std::string("mName=") + name + ","; size_t i = cur.find(k); if (i == std::string::npos) return std::string("-");
    size_t v = cur.rfind("mValue=", i); return v == std::string::npos ? std::string("-") : field_after(cur.substr(v), "mValue");
  };
  std::string bt = run_cmd("dumpsys battery 2>/dev/null");
  std::string lv = field_after(bt, "  level"), tp = field_after(bt, "  temperature"), stt = field_after(bt, "  status"),
              ac = field_after(bt, "AC powered"), usb = field_after(bt, "USB powered");
  g_bat = lv.empty() ? -1 : atoi(lv.c_str());
  char b[256];
  snprintf(b, sizeof b, "tstat=%s AP=%s BAT=%s bat=%s btemp=%s chg=%s ac=%s usb=%s", field_after(th, "Thermal Status").c_str(),
           temp_of("AP").c_str(), temp_of("BAT").c_str(), lv.empty() ? "-" : lv.c_str(), tp.empty() ? "-" : tp.c_str(),
           stt.empty() ? "-" : stt.c_str(), ac.c_str(), usb.c_str());
  return b;
}
static int battery_pct() { return g_bat; }
static double ap_of(const std::string& pw) { size_t i = pw.find("AP="); return i == std::string::npos ? -1 : atof(pw.c_str() + i + 3); }
static int tstat_of(const std::string& pw) { size_t i = pw.find("tstat="); return i == std::string::npos ? -1 : atoi(pw.c_str() + i + 6); }
static long mem_avail_mb() {
  FILE* f = fopen("/proc/meminfo", "r"); if (!f) return -1; char l[256]; long v = -1;
  while (fgets(l, sizeof l, f)) if (!strncmp(l, "MemAvailable:", 13)) { v = atol(l + 13) / 1024; break; }
  fclose(f); return v;
}
static std::string gpu_state() {
  // Exynos(Mali) 常见路径: /sys/kernel/gpu/gpu_clock、gpu_busy; 通用 kbase: /sys/class/misc/mali0/device/clock、utilization
  std::string clk = read_sys("/sys/kernel/gpu/gpu_clock"), busy = read_sys("/sys/kernel/gpu/gpu_busy");
  if (clk.empty()) clk = read_sys("/sys/class/misc/mali0/device/clock");
  if (busy.empty()) busy = read_sys("/sys/class/misc/mali0/device/utilization");
  return "gclk=" + (clk.empty() ? std::string("-") : clk) + " gbusy=" + (busy.empty() ? std::string("-") : busy);
}

struct Cmp { double one_minus_cos, maxdiff; };
template <class TY>
static Cmp cmp(const float* x, const TY* y, size_t n) {
  double xy = 0, xx = 0, yy = 0, md = 0;
  for (size_t i = 0; i < n; ++i) { double a = x[i], b = y[i]; xy += a * b; xx += a * a; yy += b * b; double d = fabs(a - b); if (!(d <= md)) md = d; }
  double c = (xx > 0 && yy > 0) ? xy / sqrt(xx * yy) : 0; return {1 - c, md};
}

// 内存保险丝(10-08 A12 实测: 448 档 fp32 GPU 建图时全机内存耗尽, 三星内核在 OOM 杀进程时 panic 整机重启,
//   重启后 adb 授权丢失要人去点)。后台线程每 100ms 读 MemAvailable, 低于门槛就记一行并 _exit 自杀(GPU 内存随进程释放)。
static std::atomic<long> g_min_mem{1L << 40};
static void mem_guard(long thr_mb) {
  for (;;) {
    long m = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (f) { char l[256]; while (fgets(l, sizeof l, f)) if (!strncmp(l, "MemAvailable:", 13)) { m = atol(l + 13) / 1024; break; } fclose(f); }
    if (m > 0 && m < g_min_mem) g_min_mem = m;
    if (m > 0 && m < thr_mb) {
      logf("mem-guard 触发: MemAvailable %ldMB < %ldMB, 探针自杀保护手机(这一档在本机装不下)", m, thr_mb);
      _exit(42);
    }
    usleep(100000);
  }
}

int main(int argc, char** argv) {
  std::string model, inp, ref, ref2, out = "abort_probe_android.txt", backend = "gpu", prec = "fp32", api = "auto", dump_dir = ".";
  int T = 0, N = 100, min_bat = 30, dump_bad = 1, threads = 4, alt = 0; double gap = 0; long guard_mb = 350; int reload = 0, verbose = 1; std::string warm_model; double warm_ap = 55, warm_max = 600, warm_min = 0; int keepwarm = 0; std::string pref = "fast", cache_dir = "";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--model") model = v; else if (k == "--inp") inp = v; else if (k == "--ref") ref = v; else if (k == "--ref2") ref2 = v;
    else if (k == "--T") T = atoi(v.c_str()); else if (k == "--n") N = atoi(v.c_str());
    else if (k == "--backend") backend = v; else if (k == "--prec") prec = v; else if (k == "--api") api = v;
    else if (k == "--out") out = v; else if (k == "--min-bat") min_bat = atoi(v.c_str());
    else if (k == "--gap") gap = atof(v.c_str()); else if (k == "--dump-bad") dump_bad = atoi(v.c_str());
    else if (k == "--dump-dir") dump_dir = v; else if (k == "--threads") threads = atoi(v.c_str());
    else if (k == "--alt") alt = atoi(v.c_str());
    else if (k == "--mem-guard") guard_mb = atol(v.c_str());
    else if (k == "--reload") reload = atoi(v.c_str());
    else if (k == "--verbose") verbose = atoi(v.c_str());
    else if (k == "--warm-model") warm_model = v; else if (k == "--warm-ap") warm_ap = atof(v.c_str());
    else if (k == "--warm-max") warm_max = atof(v.c_str()); else if (k == "--warm-min") warm_min = atof(v.c_str());
    else if (k == "--keepwarm") keepwarm = atoi(v.c_str());
    else if (k == "--pref") pref = v; else if (k == "--cache-dir") cache_dir = v;
    else { fprintf(stderr, "未知参数 %s\n", k.c_str()); return 2; }
  }
  if (model.empty() || inp.empty() || T <= 0) { fprintf(stderr, "要 --model --inp --T\n"); return 2; }
  g_log = fopen(out.c_str(), "a");
  std::thread(mem_guard, guard_mb).detach();
  std::string tz_all; thermal(&tz_all);
  logf("android-probe 开始 model=%s T=%d n=%d backend=%s prec=%s api=%s alt=%d %s mem=%ldMB %s",
       model.c_str(), T, N, backend.c_str(), prec.c_str(), api.c_str(), alt, power_state().c_str(), mem_avail_mb(), gpu_state().c_str());
  logf("thermal-zones %s", tz_all.c_str());
  const char* mali_nodes[] = {"/sys/class/misc/mali0/device/js_timeouts", "/sys/class/misc/mali0/device/js_scheduling_period",
                              "/sys/class/misc/mali0/device/reset_timeout", "/sys/class/misc/mali0/device/gpuinfo",
                              "/sys/kernel/gpu/gpu_model", "/sys/kernel/gpu/gpu_max_clock", "/sys/kernel/gpu/gpu_min_clock"};
  for (auto p : mali_nodes) { std::string s = read_sys(p); logf("sys %s = %s", p, s.empty() ? "(读不到)" : s.c_str()); }

  std::vector<float> a, b, c, d, r;
  if (!read_f16(inp + "/stem_out.f16", (size_t)T * D, a) || !read_f16(inp + "/ctx.f16", 512 * D, b) ||
      !read_f16(inp + "/mod3.f16", 3 * 6 * D, c) || !read_f16(inp + "/rope.f16", (size_t)2 * T * 64, d)) return 3;
  std::vector<_Float16> r16;   // 参考只留 fp16(省内存: 704 档 fp32 要 140MB)
  bool have_ref = false;
  if (!ref.empty() && read_f16(ref, (size_t)T * D, r)) { r16.assign(r.begin(), r.end()); have_ref = true; }
  std::vector<_Float16> r2_16;  // --alt 1 奇数次(输入 ×0.5)的参考
  if (alt && !ref2.empty() && read_f16(ref2, (size_t)T * D, r)) r2_16.assign(r.begin(), r.end());
  else if (alt) logf("⚠️--alt 1 但没给 --ref2: 奇数次只能看 same_prev, 不比参考");
  std::vector<float>().swap(r);
  if (!have_ref) logf("⚠️没有参考输出, 只比对第 0 次");

  // ── 段: --model "f1:a,c,d;f2:h,b;f3:h,c"(h = 上一段输出); 只给一个文件 = 整块 a,b,c,d ──
  struct Seg { std::string path; std::vector<char> ins; TfLiteModel* m = nullptr; TfLiteInterpreterOptions* opt = nullptr;
               TfLiteDelegate* del = nullptr; TfLiteInterpreter* it = nullptr;
               std::string cache_dir, token; char out = 'h'; };
  std::vector<Seg> segs;
  { std::string s = model; size_t p0 = 0;
    while (p0 <= s.size()) {
      size_t p1 = s.find(';', p0); std::string one = s.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0);
      if (!one.empty()) {
        Seg g; size_t c_ = one.find(':');
        g.path = one.substr(0, c_); std::string il = c_ == std::string::npos ? "a,b,c,d" : one.substr(c_ + 1);
        size_t gt = il.find('>'); if (gt != std::string::npos) { g.out = il[gt + 1]; il = il.substr(0, gt); }
        for (char ch : il) if (ch != ',') g.ins.push_back(ch);
        segs.push_back(g);
      }
      if (p1 == std::string::npos) break; p0 = p1 + 1;
    } }
  size_t n = (size_t)T * D;
  // 缓冲区按单字母命名: a b c d = 输入文件, z = 全 0, 其余(h n s …)= 前面某段的输出
  std::map<char, std::vector<float>> buf;
  buf['z'].assign(n, 0.f);
  for (auto& g : segs) if (!buf.count(g.out)) buf[g.out].assign(n, 0.f);
  auto srcbuf = [&](char k) -> std::vector<float>* { return k == 'a' ? &a : k == 'b' ? &b : k == 'c' ? &c : k == 'd' ? &d : &buf[k]; };
  auto is_const = [&](char k) { return k == 'a' || k == 'b' || k == 'c' || k == 'd' || k == 'z'; };
  auto make = [&](Seg& g) -> bool {
    g.m = TfLiteModelCreateFromFile(g.path.c_str());
    if (!g.m) { logf("模型载入失败 %s", g.path.c_str()); return false; }
    g.opt = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetErrorReporter(g.opt, reporter, nullptr);
    TfLiteInterpreterOptionsSetNumThreads(g.opt, threads);
    if (backend == "gpu") {
      TfLiteGpuDelegateOptionsV2 go = TfLiteGpuDelegateOptionsV2Default();
      // fp32 = F32 存与算; fp16 = F16 存与算; mixed = F16 存、F32 算(delegate 的 F32_F16: MAX_PRECISION 排第二位)
      go.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
      go.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
      if (prec == "fp16") { go.is_precision_loss_allowed = 1; go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY; }
      else if (prec == "mixed") {
        go.is_precision_loss_allowed = -1; go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
        go.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION; go.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      } else { go.is_precision_loss_allowed = 0; go.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION; }
      // fast = FAST_SINGLE_ANSWER(不做 kernel 调优, 建图快; 10-08 A12 上 SUSTAINED_SPEED 建一段要 113~272 s)
      go.inference_preference = pref == "sustained" ? TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED : TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
      if (!cache_dir.empty()) {   // 编译好的 GPU 程序落盘缓存, --reload 1 时第二次起建图快
        go.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
        g.cache_dir = cache_dir; g.token = "icce_" + std::to_string(std::hash<std::string>()(g.path + prec + api + pref));
        go.serialization_dir = g.cache_dir.c_str(); go.model_token = g.token.c_str();
      }
      if (api == "cl") go.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
      if (api == "gl") go.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
      go.max_delegated_partitions = 1;
      g.del = TfLiteGpuDelegateV2Create(&go);
      if (!g.del) { logf("GPU delegate 创建失败"); return false; }
      TfLiteInterpreterOptionsAddDelegate(g.opt, g.del);
    }
    g.it = TfLiteInterpreterCreate(g.m, g.opt);
    if (!g.it) { logf("解释器创建失败 %s(GPU delegate 没接上? 看上面 tflite-msg)", g.path.c_str()); return false; }
    if (TfLiteInterpreterAllocateTensors(g.it) != kTfLiteOk) { logf("AllocateTensors 失败"); return false; }
    int ni = TfLiteInterpreterGetInputTensorCount(g.it);
    if (ni != (int)g.ins.size()) { logf("%s 输入 %d 个, 段说明给了 %zu 个", g.path.c_str(), ni, g.ins.size()); return false; }
    for (int i = 0; i < ni; ++i) {
      TfLiteTensor* t = TfLiteInterpreterGetInputTensor(g.it, i);
      std::vector<float>* sb = srcbuf(g.ins[i]);
      if (TfLiteTensorByteSize(t) != sb->size() * 4) { logf("输入 %d(%c) 字节 %zu ≠ %zu", i, g.ins[i], TfLiteTensorByteSize(t), sb->size() * 4); return false; }
      if (is_const(g.ins[i])) TfLiteTensorCopyFromBuffer(t, sb->data(), sb->size() * 4);   // 常量输入一次拷好
    }
    if (TfLiteTensorByteSize(TfLiteInterpreterGetOutputTensor(g.it, 0)) != n * 4) { logf("输出大小不对"); return false; }
    return true;
  };
  auto drop = [&](Seg& g) {
    if (g.it) TfLiteInterpreterDelete(g.it); if (g.del) TfLiteGpuDelegateV2Delete(g.del);
    if (g.opt) TfLiteInterpreterOptionsDelete(g.opt); if (g.m) TfLiteModelDelete(g.m);
    g.it = nullptr; g.del = nullptr; g.opt = nullptr; g.m = nullptr;
  };
  logf("段数 %zu reload=%d pref=%s cache=%s", segs.size(), reload, pref.c_str(), cache_dir.empty() ? "-" : cache_dir.c_str());
  if (!reload) {
    for (auto& g : segs) {
      double t0 = mono();
      if (!make(g)) return 6;
      logf("建图 %s %.1fs 期间最低 mem=%ldMB 现 mem=%ldMB", g.path.c_str(), mono() - t0, (long)g_min_mem, mem_avail_mb());
    }
  }
  // ── GPU 空转预热 / 保温(热机实验): 反复调一个不带权重的注意力模型(输入全 0), 只为让 GPU 满载发热 ──
  Seg W; W.path = warm_model;
  auto warm_make = [&]() -> bool {
    if (warm_model.empty()) return false;
    if (W.it) return true;
    if (!make(W)) return false;
    for (int i = 0; i < TfLiteInterpreterGetInputTensorCount(W.it); ++i) {
      TfLiteTensor* t = TfLiteInterpreterGetInputTensor(W.it, i); memset(TfLiteTensorData(t), 0, TfLiteTensorByteSize(t));
    }
    return true;
  };
  if (!warm_model.empty()) {
    // make() 按段说明核对输入: 预热模型输入个数不定, 临时塞够 'z'
    { TfLiteModel* m_ = TfLiteModelCreateFromFile(warm_model.c_str()); TfLiteInterpreterOptions* o_ = TfLiteInterpreterOptionsCreate();
      TfLiteInterpreter* i_ = m_ ? TfLiteInterpreterCreate(m_, o_) : nullptr;
      int k_ = i_ ? TfLiteInterpreterGetInputTensorCount(i_) : 0; W.ins.assign(k_, 'z');
      if (i_) TfLiteInterpreterDelete(i_); TfLiteInterpreterOptionsDelete(o_); if (m_) TfLiteModelDelete(m_); }
    double w0 = mono(); int wn = 0;
    if (!warm_make()) { logf("预热模型建图失败, 不预热"); }
    else for (;;) {
      std::string pw = power_state();
      double el = mono() - w0;
      if ((el >= warm_min && (ap_of(pw) >= warm_ap || tstat_of(pw) > 0)) || el >= warm_max) {
        logf("预热结束 %.0fs %d 次 %s mem=%ld", el, wn, pw.c_str(), mem_avail_mb()); break; }
      double s_ = mono(); TfLiteStatus ws = TfLiteInterpreterInvoke(W.it); wn++;
      if (wn % 5 == 1) logf("预热 #%d %.1fs st=%d %s mem=%ld", wn, mono() - s_, (int)ws, pw.c_str(), mem_avail_mb());
    }
    if (reload || !keepwarm) drop(W);
  }
  std::vector<_Float16> first(n);
  uint64_t prev_hash = 0;
  const float SENT = -12345.678f;
  int dumped = 0, nbad_calls = 0;
  std::vector<float> outbuf(n);
  float* op = outbuf.data();

  for (int iter = 0; iter < N; ++iter) {
    std::string pw0 = power_state();
    int bat = battery_pct();
    if (bat >= 0 && bat < min_bat) { logf("电量 %d%% < %d%%, 停", bat, min_bat); break; }
    bool half = alt && (iter & 1);
    std::string th0 = thermal(nullptr); long mem0 = mem_avail_mb();
    int err0 = g_err_count;
    TfLiteStatus st = kTfLiteOk; double dt = 0, tload = 0; std::string segdt, segsent;
    for (size_t si = 0; si < segs.size(); ++si) {
      Seg& g = segs[si];
      if (reload) { double tl = mono(); long m0 = mem_avail_mb(); g_min_mem = 1L << 40;
        if (!make(g)) { logf("第 %d 次 第 %zu 段建图失败, 停", iter, si); goto done; } tload += mono() - tl;
        if (verbose) logf("  i=%d 段%zu 建图 %.1fs mem %ld→%ld 最低 %ld", iter, si, mono() - tl, m0, mem_avail_mb(), (long)g_min_mem); }
      for (size_t i = 0; i < g.ins.size(); ++i) {
        TfLiteTensor* t = TfLiteInterpreterGetInputTensor(g.it, i);
        if (!is_const(g.ins[i])) TfLiteTensorCopyFromBuffer(t, srcbuf(g.ins[i])->data(), n * 4);
        else if (g.ins[i] == 'a' && alt) {   // 奇数次 hidden 输入 ×0.5
          float* ain = (float*)TfLiteTensorData(t);
          for (size_t k = 0; k < n; ++k) ain[k] = half ? a[k] * 0.5f : a[k];
        }
      }
      float* go_ = (float*)TfLiteTensorData(TfLiteInterpreterGetOutputTensor(g.it, 0));
      for (size_t k = 0; k < n; ++k) go_[k] = SENT;   // 哨兵: 读回没发生就还在
      double s = mono();
      TfLiteStatus st1 = TfLiteInterpreterInvoke(g.it);
      double d1 = mono() - s; dt += d1;
      if (st1 != kTfLiteOk) st = st1;
      size_t sc = 0; for (size_t k = 0; k < n; ++k) if (go_[k] == SENT) sc++;
      char b_[64]; snprintf(b_, sizeof b_, "%s%.2f", si ? "," : "", d1); segdt += b_;
      snprintf(b_, sizeof b_, "%s%zu", si ? "," : "", sc); segsent += b_;
      memcpy(si + 1 < segs.size() ? buf[g.out].data() : op, go_, n * 4);
      if (verbose) logf("  i=%d 段%zu 推理 %.2fs mem=%ld 最低 %ld", iter, si, d1, mem_avail_mb(), (long)g_min_mem);
      if (reload) { drop(g); if (verbose) logf("  i=%d 段%zu 释放后 mem=%ld", iter, si, mem_avail_mb()); }
    }
    {
    int nerr = g_err_count - err0;
    size_t zero = 0, naninf = 0, sent = 0; double ss = 0, amax = 0;
    for (size_t k = 0; k < n; ++k) {
      float v = op[k];
      if (v == SENT) { sent++; continue; }
      if (!isfinite(v)) { naninf++; continue; }
      if (v == 0.f) zero++;
      ss += (double)v * v; amax = std::max(amax, (double)fabsf(v));
    }
    double rms = sqrt(ss / n);
    Cmp cr{NAN, NAN}, cf{0, 0};
    long badtok = -1, fb = -1, lb = -1, tail64 = -1; double maxrel = 0;
    const _Float16* rr = half ? (r2_16.empty() ? nullptr : r2_16.data()) : (have_ref ? r16.data() : nullptr);
    if (rr) {
      cr = cmp(op, rr, n);
      badtok = 0; tail64 = 0;
      for (int t = 0; t < T; ++t) {
        double num = 0, den = 0;
        for (int j = 0; j < D; ++j) { double x = op[(size_t)t * D + j], y = (double)rr[(size_t)t * D + j]; num += (x - y) * (x - y); den += y * y; }
        double rel = (den > 0) ? sqrt(num / den) : (num > 0 ? 1e9 : 0);
        if (!(rel <= 0.01)) { badtok++; if (fb < 0) fb = t; lb = t; if (t >= T - 64) tail64++; }
        if (!(rel <= maxrel)) maxrel = rel;
      }
    }
    if (iter == 0) for (size_t k = 0; k < n; ++k) first[k] = (_Float16)op[k];
    else if (!half) cf = cmp(op, first.data(), n);
    uint64_t hsh = 1469598103934665603ull; const uint8_t* bp = (const uint8_t*)op;
    for (size_t k = 0; k < n * 4; ++k) { hsh ^= bp[k]; hsh *= 1099511628211ull; }
    bool same_prev = iter > 0 && hsh == prev_hash;   // 与上一次输出逐位相同(alt 模式下 = 这次没真算)
    prev_hash = hsh;
    bool bad = (st != kTfLiteOk) || sent || naninf || (rr && (cr.one_minus_cos > 1e-3 || badtok > 0));
    if (bad) nbad_calls++;
    logf("android-probe i=%d %s th0=[%s] th=[%s] pw0=[%s] pw=[%s] mem0=%ld mem=%ld %s dt=%.2fs segdt=%s tload=%.1f threw=%s msgs=%d zero=%.4f naninf=%zu sentinel=%zu segsent=%s "
         "minmem=%ld rms=%.4f amax=%.1f ref_1mcos=%.3e ref_maxdiff=%.4g first_1mcos=%.3e same_prev=%d badtok=%ld/%d first=%ld last=%ld tail64=%ld maxrel=%.3g bad=%d",
         iter, half ? "in=half" : "in=full", th0.c_str(), thermal(nullptr).c_str(), pw0.c_str(), power_state().c_str(), mem0, mem_avail_mb(),
         gpu_state().c_str(), dt, segdt.c_str(), tload, st == kTfLiteOk ? "-" : std::to_string((int)st).c_str(), nerr, (double)zero / n, naninf, sent,
         segsent.c_str(), (long)g_min_mem, rms, amax, cr.one_minus_cos, cr.maxdiff, cf.one_minus_cos, (int)same_prev, badtok, T, fb, lb, tail64, maxrel, (int)bad);
    if (bad && dumped < dump_bad) {
      std::string p = dump_dir + "/bad_out_i" + std::to_string(iter) + ".f16";
      FILE* f = fopen(p.c_str(), "wb");
      if (f) { std::vector<_Float16> hb(n); for (size_t k = 0; k < n; ++k) hb[k] = (_Float16)op[k]; fwrite(hb.data(), 2, n, f); fclose(f); dumped++; logf("存坏输出 %s", p.c_str()); }
    }
    }
    if (keepwarm > 0 && warm_make()) {   // 保温: 调用之间再空转 keepwarm 次(不留空档)
      double s_ = mono(); for (int k = 0; k < keepwarm; ++k) TfLiteInterpreterInvoke(W.it);
      if (verbose) logf("  i=%d 保温 %d 次 %.1fs", iter, keepwarm, mono() - s_);
      if (reload) drop(W);
    }
    if (gap > 0) usleep((useconds_t)(gap * 1e6));
  }
done:
  logf("android-probe 结束 坏 %d 次 %s mem=%ldMB", nbad_calls, power_state().c_str(), mem_avail_mb());
  for (auto& g : segs) drop(g);
  drop(W);
  if (g_log) fclose(g_log);
  return 0;
}
