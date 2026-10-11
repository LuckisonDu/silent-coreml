#!/bin/bash
# 交叉编译安卓探针(arm64-v8a, API 26)。NDK r27c 与 LiteRT 1.4.2 AAR(litert + litert-gpu, 经典 TFLite C API + GPU delegate V2)
# 都放在 $HOME/tools/。产物 build/icce_gpu_probe + build/lib/*.so(推到手机 /data/local/tmp/icce/)。
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=${TOOLS:-$HOME/tools}
NDK=${NDK:-$TOOLS/android-ndk-r27c}
AAR=$TOOLS/litert_aar
X=$AAR/x
if [ ! -f "$X/jni/arm64-v8a/libtensorflowlite_gpu_jni.so" ]; then
  mkdir -p "$AAR" && cd "$AAR"
  for a in litert litert-gpu; do
    [ -f $a-1.4.2.aar ] || curl -sfLO https://dl.google.com/android/maven2/com/google/ai/edge/litert/$a/1.4.2/$a-1.4.2.aar
  done
  mkdir -p x && cd x && for f in ../*.aar; do unzip -qo $f 'jni/arm64-v8a/*' 'headers/*'; done
fi
mkdir -p "$HERE/build/lib"
cp "$X/jni/arm64-v8a/libtensorflowlite_jni.so" "$X/jni/arm64-v8a/libtensorflowlite_gpu_jni.so" "$HERE/build/lib/"
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++
"$CXX" -O2 -std=c++17 -I"$X/headers" "$HERE/icce_gpu_probe.cc" -o "$HERE/build/icce_gpu_probe" \
  -L"$HERE/build/lib" -ltensorflowlite_jni -ltensorflowlite_gpu_jni -llog -Wl,-rpath,'$ORIGIN/lib' -static-libstdc++
file "$HERE/build/icce_gpu_probe"
echo "OK → $HERE/build"
# 长任务探针(单个 BATCH_MATMUL 超过 Mali 硬停时限, 见 icce_longjob_probe.cc 头注)
"$CXX" -O2 -std=c++17 -I"$X/headers" "$HERE/icce_longjob_probe.cc" -o "$HERE/build/icce_longjob_probe" \
  -L"$HERE/build/lib" -ltensorflowlite_jni -ltensorflowlite_gpu_jni -llog -Wl,-rpath,'$ORIGIN/lib' -static-libstdc++
file "$HERE/build/icce_longjob_probe"
