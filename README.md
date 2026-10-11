# Self-Checking Video Generation on iPhone: Detecting and Recovering from Silent Output Corruption

Companion code and data for the ICCE 2027 submission (paper version v60). The paper's conclusion states that we release the detector, the probe and the per-call logs; this package contains them, plus the scripts that recompute the paper's numbers from those logs.

## Contents

| Directory | What it contains |
|---|---|
| `probe/` | On-device code. `SegTest.swift`: the iOS probe (`abortProbe`). `FWPipeline_gpucheck_excerpt.swift`: the per-block check used for the full clips (`--gpu-check=tail` / `stride`), excerpted from the app. `android/`: the LiteRT probe for the Galaxy A12 (the Wan block-to-LiteRT exporter is not included; the synthetic long-job model builder `build_longjob_tflite.py` is). `mac_abort_probe.py`: the Mac M4 control. |
| `analysis/` | Python scripts that recompute every logged number in the paper (see the table below). |
| `logs/` | Raw logs, one line per call: 95 iOS probe runs (`<run>/abort_probe.txt`, plus `args.txt` with the probe flags where recorded), full-clip traces (`e2e/`), Galaxy A12 runs (`android/`), Mac M4 control (`mac_m4/`). `runs.tsv` gives the time each run's log was copied off the phone. |
| `derived_v60/` | The derived files as they stood for paper v60 (`numbers.tex` holds the LaTeX number macros, `table_tests.tex` the test table, and the JSON files hold the intermediate statistics). They are kept unchanged so that the recomputed output can be compared with them. |
| `tools/` | `make_view.py` (snapshot views, see below) and `compare.py`. |
| `reproduce.sh` | Runs the full recomputation and the comparison. |

### Devices

| Name in paper and logs | Device | SoC | OS during the probes |
|---|---|---|---|
| Phone A (`phoneA`) | iPhone 12, 4 GB | A14 | iOS 26.5.2 |
| Phone B (`phoneB`) | iPhone 12, 4 GB | A14 | iOS 26.3.1; upgraded to iOS 27.0.1 for the runs named `probephoneBosios27*`, the two tail-checked clips `phoneB_s104_tail` / `phoneB_s105_tail` |
| Phone C (`phoneC`) | iPhone 12, 4 GB | A14 | iOS 26.6.2 |
| iPhone 18 Pro (`iphone18pro`) | iPhone 18 Pro | A20 | iOS 27.0.1 |
| Galaxy A12 (`android/a12_*`) | Samsung Galaxy A12 | Exynos 850, Mali-G52 MP1 | Android 11, LiteRT 1.4.2 GPU delegate (OpenCL) |
| Mac M4 (`mac_m4/`) | Mac mini M4 | M4 | macOS 26 |

Run directories are named `probe<device><experiment>_<MMDD>_<HHMM>`. A few early runs have shorter names (`probe2_…_phoneA`, `probe3_…_phoneA`, `probephoneA4_…`, `probephoneA5split_…` are Phone A; `probephoneB2_…`, `probephoneB_…` are Phone B; `probephoneC_…`, `probephoneC61_…` are Phone C).

### Not included

* **Model weights.** Wan2.2-TI2V-5B Turbo is a third-party model. Its Core ML block packages and the exporter are not included. A verifier is the same block with the same weights, exported so that it outputs only the K = 64 query rows it checks: the last 64 tokens (tail) or rows p_k = floor((k+0.5)T/K) (strided); every other part of the block is unchanged.
* **Input tensors and references.** These were recorded on the phone during a private generation, so the paper's statement that tensors were "saved in full" (`dump_analysis.json`) and the per-token error figure (`fig/truncation.pdf`) cannot be recomputed from this package. `derived_v60/dump_analysis.json` is included for reference.
* **Images and videos.** The package contains no first frames, clips or tensor dumps. For the full clips we include `auto_output.framemd5` instead, which is the output of `ffmpeg -f framemd5` on each clip. It is enough to check that two clips are frame-identical but not to compute PSNR. The PSNR values (the 47/46 dB of the iOS 27 tail-checked clips, and the PSNR fields of `e2e_compare.json`) are therefore given only in `derived_v60/`.
* Each per-clip trace keeps only the segment for that clip. The phone appends every clip to one trace file, and the scripts read only the last segment.

## Recomputing the numbers

```bash
# python3 with numpy and scipy; matplotlib only for the figures
bash reproduce.sh                       # or: PY=/path/to/python bash reproduce.sh
python3 analysis/fig_v25.py views/s1008 figs/   # Fig. rates_char / rates_mitig
```

`reproduce.sh` writes only into `views/`. Its last step compares the regenerated files with `derived_v60/`. On our machine, all files are identical, including every macro of `numbers.tex` and `table_tests.tex`. The one exception is the `e2e` entry of `paper_stats_v3.json`. That entry now also lists the tail-checked clips, and because no videos are shipped, its PSNR fields are `None`. No paper number is taken from it.

**Snapshots.** The derived files were written while data were still being collected, so each one was computed from the runs that existed at the time it was written. `tools/make_view.py "<time>" views/<name>` builds a view of `logs/` (symlinks) that contains only the runs copied off the phone by `<time>` (Pacific time). `reproduce.sh` uses these cut-offs:

| View | Cut-off | Used for |
|---|---|---|
| `s1007a` | 10-07 07:44 | `corrupt_structure.json` (truncation shape, `\ntr…`) |
| `s1007b` | 10-07 11:47 | `tail_overhead.json` (tail-check cost) |
| `s1007c` | 10-07 15:29 | `abort_log_link.json` (abort record vs. corrupted output) |
| `s1007d` | 10-07 19:20 | `cluster_stats.json` (moving-block bootstrap, table) |
| `s1008` | 10-08 08:24 | `paper_stats_v3.json` → `numbers.tex`, `table_tests.tex` |
| `s1009a` | 10-09 11:46 | `check_k_curve.json` |
| `s1009b` | 10-09 12:57 | `stat_detector.json` |
| `s1009c` | 10-09 13:10 | `check_positions.json`, `holdout_injection.json` (= all runs in `logs/`) |

### Where each number comes from

| Paper statement | Script (output) |
|---|---|
| main set 93/350 (31/150, 10/100, 52/100); thermal strata; gating; reboot series; recovery policies; Klein; LTX; A20 natural load; all `\n…` macros | `paper_stats.py` → `gen_numbers.py` (`numbers.tex`), `gen_table.py` |
| truncation shape (`\ntrN`, suffix fraction, first wrong token) | `corrupt_structure.py` |
| abort record in the call window (`\nalCR/\nalCN`, `\nalOR/\nalON`) | `abort_log_link.py` |
| tail variant on the phone (`\ntcCaught/\ntcWrong`, false alarms, cost) | `paper_stats.py`, `tail_overhead.py` |
| 1047 recorded non-zero corrupted outputs (814 A14 whole-block, 90 A14 split, 143 A20), strided 1047/1047 vs. tail; expected misses | `check_positions.py` |
| hold-out by time 320/320 (tail 307); by phone 514/514 (tail 497); coverage bound 179 tokens; scattered errors 2.8% / 43%; 1430-token span at density 0.2 → 0.83 | `holdout_injection.py --paper-v60` (see note) |
| tail misses at every K from 8 to 256; strided complete from K = 64 | `check_k_curve.py` |
| output-statistics pre-filter (3% / 7% bands) | `stat_detector.py` |
| paired runs: 27/27 strided, 26/27 tail, false alarms 5/45 and 4/45 (iPhone 18 Pro); 47/47 both, 0/5 (hot Phone A) | `paired_stride.py` |
| 628 Metal command buffers all `InnocentVictim`; 63/136 Core ML calls corrupted, none threw, none without an abort record; tail-only induced run 44/48, 3/51 false alarms | `gpuhang_crosslayer.py` |
| full clips: unhandled aborts, reruns, 31 vs. 28 min, pixel-identical arms | `e2e_tally.py`, `e2e_compare.py`, `paper_stats.py` |
| tail-checked clips (`\nteChecks`, `\nteFlag`, 43/163, 61/181, 54 / 62 min, frame-identical to the log-checked clip) | `tail_e2e.py` |
| Galaxy A12 0/48 and 0/60 long jobs | `android_counts.py` (per run: `tally_android.py`) |
| Mac M4: 40/40 bit-identical outputs | `logs/mac_m4/mac_run1.txt` (all 40 lines have the same `rms` and `cos`) |
| split / gap / keep-warm runs per segment | `split_tally.py`, `gap_tally.py` |

**Note on the hold-out by phone.** Paper v60 assigned runs to phones by directory name. The early Phone A runs `probephoneA4_*` and `probephoneA5split_*` carried no device name at the time, so they were counted in the hold-out half. `--paper-v60` reproduces that count: 514, with the tail catching 497. When those runs are assigned to Phone A, which is how `holdout_injection.py` runs without the flag, the hold-out half has 489 outputs. The tail catches 474 of them (321 A14 whole-block; 25 A14 split, of which the tail catches 23; 143 A20, of which the tail catches 130). The strided check still catches all 489.

**Note on the Galaxy A12 "48".** The 48 block calls are all the mixed-precision block-0 calls of 10-08: 35 on a cool phone and 13 on a heated one.

## Reading the logs

Every probe line starts with a UTC timestamp. Some markers and field values are in Chinese:

* `abort-probe 开始 档=<tier> iters=N … 发热 <t> ui压力=<bool>`: start of a run (`档` = model export, `发热` = `ProcessInfo.thermalState` at start, 0 nominal … 3 critical, `ui压力` = 30 Hz interface refresh).
* `abort-probe i=<k> …`: one Core ML call. `th` / `th0` thermal state after / before the call; `bat`, `pwr`, `lpm` battery state; `dt` call time; `threw` error raised (`-` = none); `oslog` = GPU abort records in the call's window (`GPU 命令缓冲被系统掐断 n 次(影响交互 a / 连带丢弃 b)` = n command-buffer aborts, a `ImpactingInteractivity`, b `InnocentVictim`; `-` = none); `zero`, `naninf`, `rms`; `ref_1mcos` = 1 − cos to the reference (corrupted if > 1e-3); `badtok=<wrong>/<T> first= last=` per-token errors (token wrong if its norm differs by > 1%); `tailchk` / `tail2chk` = 1 − cos of the tail / strided verifier (flag if > 1e-4); `tdt`, `tload` verifier time and load time; `seg=` split-call segment.
* `abort-probe 卡核 #n … 结果 …`: one long-running Metal command buffer submitted by `--abort-probe-gpuhang` and the error it completed with.
* `abort-recover ep=…`: one recovery episode (`pol` = imm / wait / reload).
* Full-clip traces (`e2e/*/trace.txt`): `末尾校验` / `分散校验 块NN 1-cos x` = tail / strided check of block NN; `作废` = output discarded and block rerun after a wait; `未处理` = abort record left unhandled (no-check arm).

Container paths in the logs are replaced by `<AppContainer>`, and device and host names by the code names above.

## Probe usage (iOS)

The probe is part of the evaluation app and runs from command-line arguments (see the header of `abortProbe` in `SegTest.swift`), for example:

```
--abort-probe-iters=100 --abort-probe-ui --abort-probe-tail=704x1280f32t64 --abort-probe-tail2=704x1280f32s64:stride64
```

`--abort-probe-tail` / `--abort-probe-tail2` run a tail (`t64`) or strided (`s64`, export `FW_SPLIT_AT=rows FW_ROWS=stride64`) verifier after every call; `--abort-probe-gpuhang=<n>` keeps a background long-running Metal kernel to induce GPU recovery; `--abort-probe-metalsent` submits short sentinel command buffers and logs their status; `--abort-probe-recover=imm,wait,reload` runs recovery episodes. The strided positions are p_k = floor((k + 0.5) · T / K), K = 64, the same formula as in `check_positions.py` and in `tailCheck()` of the excerpt.
