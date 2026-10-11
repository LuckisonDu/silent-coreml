#!/bin/bash
# Recompute the paper's numbers from the per-call logs and compare with the frozen v60 files (derived_v60/).
# Needs python3 with numpy and scipy (matplotlib only for the figure scripts). Writes only into views/.
# Usage: bash reproduce.sh            (PY=/path/to/python bash reproduce.sh to pick an interpreter)
set -euo pipefail
cd "$(dirname "$0")"; PY=${PY:-python3}; A=analysis; V=views
rm -rf $V; mkdir -p $V
mk() { $PY tools/make_view.py "$1" $V/$2; }
# The derived files were written while data were still coming in; each is recomputed on the data available when it was written.
mk "2026-10-07 07:44" s1007a; mk "2026-10-07 11:47" s1007b; mk "2026-10-07 15:29" s1007c; mk "2026-10-07 19:20" s1007d
mk "2026-10-08 08:24" s1008;  mk "2026-10-09 11:46" s1009a; mk "2026-10-09 12:57" s1009b; mk "2026-10-09 13:10" s1009c
cp -r $V/s1009c $V/s1009c_byphone_fixed 2>/dev/null || true
q() { echo "+ $*"; "$@" > "$V/last.log" 2>&1 || { cat "$V/last.log"; exit 1; }; }
q $PY $A/e2e_tally.py $V/s1008/e2e;  q $PY $A/e2e_compare.py $V/s1008/e2e
q $PY $A/corrupt_structure.py $V/s1007a; q $PY $A/tail_overhead.py $V/s1007b; q $PY $A/abort_log_link.py $V/s1007c; q $PY $A/cluster_stats.py $V/s1007d
q $PY $A/tail_e2e.py $V/s1009c
cp $V/s1007a/corrupt_structure.json $V/s1007b/tail_overhead.json $V/s1007c/abort_log_link.json $V/s1007d/cluster_stats.json $V/s1009c/tail_e2e.json $V/s1008/
q $PY $A/paper_stats.py $V/s1008
q $PY $A/gen_numbers.py $V/s1008 $V/numbers.tex; q $PY $A/gen_table.py $V/s1008 $V/table_tests.tex
q $PY $A/check_k_curve.py $V/s1009a; q $PY $A/stat_detector.py $V/s1009b
q $PY $A/check_positions.py $V/s1009c; q $PY $A/holdout_injection.py $V/s1009c --paper-v60; q $PY $A/independence.py $V/s1009c
q $PY $A/holdout_injection.py $V/s1009c_byphone_fixed
echo "== paired on-phone runs (both verifiers on every call)"
$PY $A/paired_stride.py logs/probeiphone18prostride_1009_1020/abort_probe.txt
$PY $A/paired_stride.py logs/probephoneAstride_1009_1111/abort_probe.txt
echo "== induced GPU recovery on the iPhone 18 Pro: Metal vs Core ML"
(cd $V && $PY ../$A/gpuhang_crosslayer.py ../logs/probeiphone18proosgpuhang_1009_0102/abort_probe.txt ../logs/probeiphone18proosgpuhangtail3_1009_0704/abort_probe.txt)
echo "== Galaxy A12"; $PY $A/android_counts.py logs/android
echo "== comparison with derived_v60/"; $PY tools/compare.py
