"""Compare regenerated outputs in views/ with the frozen files of paper v60 in derived_v60/. Usage: python3 tools/compare.py"""
import json, os, filecmp
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); V = os.path.join(here, 'views'); D = os.path.join(here, 'derived_v60')
pairs = [('s1007a/corrupt_structure.json', 'corrupt_structure.json'), ('s1007b/tail_overhead.json', 'tail_overhead.json'),
         ('s1007c/abort_log_link.json', 'abort_log_link.json'), ('s1007d/cluster_stats.json', 'cluster_stats.json'),
         ('s1008/paper_stats_v3.json', 'paper_stats_v3.json'), ('s1009c/tail_e2e.json', 'tail_e2e.json'),
         ('s1009a/check_k_curve.json', 'check_k_curve.json'), ('s1009b/stat_detector.json', 'stat_detector.json'),
         ('s1009c/check_positions.json', 'check_positions.json'), ('s1009c/holdout_injection.json', 'holdout_injection.json')]
for a, b in pairs:
    x, y = json.load(open(os.path.join(V, a))), json.load(open(os.path.join(D, b)))
    if x == y: print(f'IDENTICAL  {b}'); continue
    d = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k)); print(f'DIFFERS    {b}: keys {d}')
for f in ('numbers.tex', 'table_tests.tex'):
    print(('IDENTICAL  ' if filecmp.cmp(os.path.join(V, f), os.path.join(D, f), shallow=False) else 'DIFFERS    ') + f)
