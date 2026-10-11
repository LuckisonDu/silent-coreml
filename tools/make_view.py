"""Build a snapshot view of logs/: symlink every run directory whose log was pulled from the phone at or before CUTOFF
(column pulled_at_PDT of logs/runs.tsv), plus a copy of logs/e2e. The paper's derived files were computed while data were still
being collected, so each one corresponds to the data available at the time it was written (see README, "Snapshots").
Usage: python3 tools/make_view.py "2026-10-08 08:24" views/s1008   (run from the package root)"""
import os, sys
cut, out = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__)); logs = os.path.join(here, '..', 'logs')
os.makedirs(out, exist_ok=True); n = 0
for line in open(os.path.join(logs, 'runs.tsv')).read().splitlines()[1:]:
    d, t = line.split('\t')
    if t <= cut:
        dst = os.path.join(out, d)
        if not os.path.lexists(dst): os.symlink(os.path.abspath(os.path.join(logs, d)), dst)
        n += 1
import shutil   # e2e is copied (scripts write their json next to it), run directories are only symlinked
if not os.path.exists(os.path.join(out, 'e2e')): shutil.copytree(os.path.join(logs, 'e2e'), os.path.join(out, 'e2e'))
print(f'{n} run directories <= {cut} -> {out}')
