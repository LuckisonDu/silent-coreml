"""Galaxy A12 (Mali-G52, LiteRT GPU delegate) counts used in the paper (Table "decide", Limitations): "0/48" and "0/60 long".
  block calls : Wan block 0, 448x800, fp16-storage/fp32-compute ("mixed") runs of 10-08: a12_448_gpumixed_s1r1_1008_2016 (2),
                a12_m448_gpumixed_s1r1_1008_2022 (25), a12_m448L_gpumixed_s1r1_1008_2118 (8) = 35 cool, a12_m448_gpumixed_s1r1_1008_2203 (13) heated.
                NOTE: the 48 are all 10-08 mixed-precision block calls (35 cool + 13 heated), not 48 heated calls.
  long jobs   : single CONV_2D whose one GPU dispatch takes 1.5-99 s: a12_longjob_mixed_1009_1100 (40, cool) + a12_longjob_mixed_hot_1009_1231 (20, heated).
A call counts as corrupted if bad=1 with no throw / sentinel / NaN (same rule as tally_android.py).
Usage: python3 android_counts.py logs/android"""
import re, sys, glob, os
R = sys.argv[1]
kv = lambda l, k: (re.search(rf"\b{k}=([^\s\]]+)", l) or [None, None])[1]
def tally(files, tag):
    rows = [l for f in files for l in open(f, errors='ignore') if tag in l and ' i=' in l and 'dt=' in l]
    bad = sum(kv(l, 'bad') == '1' for l in rows); thr = sum(kv(l, 'threw') not in (None, '-') for l in rows)
    dts = [float(kv(l, 'dt').rstrip('s')) for l in rows]
    return dict(calls=len(rows), corrupted_or_bad=bad, threw=thr, dt_s=(min(dts), max(dts)) if dts else None)
blk = {'cool': ['a12_448_gpumixed_s1r1_1008_2016', 'a12_m448_gpumixed_s1r1_1008_2022', 'a12_m448L_gpumixed_s1r1_1008_2118'], 'heated': ['a12_m448_gpumixed_s1r1_1008_2203']}
for k, ds in blk.items(): print('block', k, tally([f'{R}/{d}/abort_probe_android.txt' for d in ds], 'android-probe i='))
print('block total (paper "48")', tally([f'{R}/{d}/abort_probe_android.txt' for ds in blk.values() for d in ds], 'android-probe i='))
lj = {'cool': 'a12_longjob_mixed_1009_1100', 'heated': 'a12_longjob_mixed_hot_1009_1231'}
for k, d in lj.items(): print('longjob', k, tally(sorted(glob.glob(f'{R}/{d}/longjob_*.txt')), 'longjob i='))
print('longjob total (paper "60")', tally([f for d in lj.values() for f in sorted(glob.glob(f'{R}/{d}/longjob_*.txt'))], 'longjob i='))
