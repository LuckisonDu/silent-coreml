"""ICCE v25 汇总表 paper/table_tests.tex：主要比较的次数、Fisher p、Holm 校正后 p、按整轮重抽样的差值区间。只读统计文件。
用法: python3 gen_table.py logs table_tests.tex"""
import sys, json
ROOT, OUT = sys.argv[1], sys.argv[2]
S = json.load(open(f'{ROOT}/paper_stats_v3.json')); C = json.load(open(f'{ROOT}/cluster_stats.json')); H = S['holm']
f = lambda k: f"{S[k]['wrong']}/{S[k]['n']}"
def p(x):
    if x >= 0.01: return f'{x:.2g}'
    m, e = f'{x:.0e}'.split('e'); return f'${m}\\times10^{{{int(e)}}}$'
def bs(k):
    if k not in C: return '---'
    d = C[k]; m = lambda x: f"{x:g}".replace('-', '$-$'); return f"{m(d['diff'])} ({m(d['ci'][0])}--{m(d['ci'][1])})"
RL = S['recovery_phoneC']
rows = [
    ('Gating, A: ungated vs gated', f('thermalgate_ungated'), f('thermalgate_gated'), 'gating_A', 'thermalgate_A_ungated_minus_gated_L5'),
    ('Gating, B', f('thermalgate_B_ungated'), f('thermalgate_B_gated'), 'gating_B', 'thermalgate_B_ungated_minus_gated_L5'),
    ('Gating, C', f('thermalgate_C_ungated'), f('thermalgate_C_gated'), 'gating_C', 'thermalgate_C_ungated_minus_gated_L5'),
    ('Split, A: 1 call vs 3', f('split_phoneA_whole'), f('split_phoneA_3seg'), 'split_A', 'split_A_whole_minus_3seg_L5'),
    ('Split, A (hotter)', f('keepwarm_A_whole'), f('keepwarm_A_split_gap0'), 'split_A_rep', 'split_A_hot_whole_minus_3seg_L5'),
    ('Split, B', f('gap_phoneB_whole'), f('gap_phoneB_3seg_gap0'), 'split_B', 'split_B_whole_minus_3seg_L5'),
    ('Pause, B: 3\\,s vs 0\\,s', f('gap_phoneB_3seg_gap3'), f('gap_phoneB_3seg_gap0'), 'gap_B', 'gap_B_3s_minus_0s_L5'),
    ('Pause, A: 3\\,s vs 0\\,s', f('keepwarm_A_split_gap3_idle'), f('keepwarm_A_split_gap0'), 'gap_A', 'gap_A_3s_minus_0s_L5'),
    ('Pause idle vs busy, A', f('keepwarm_A_split_gap3_idle'), f('keepwarm_A_split_gap3_warm'), 'warm_vs_idle_A', 'warm_A_idle_minus_busy_L5'),
    ('Chunks, B: 8 vs 32', f('qchunk_phoneB_q8'), f('qchunk_phoneB_q32'), 'qchunk_B', 'qchunk_B_q8_minus_q32_L5'),
    ('Recovery, C: imm.\\ vs wait$^\\ast$', f"{RL['imm']['n'] - RL['imm']['final_ok']}/{RL['imm']['n']}", f"{RL['wait']['n'] - RL['wait']['final_ok']}/{RL['wait']['n']}", 'recovery_C', None),
]
L = ['\\begin{tabular}{lcccc}', '\\toprule', 'comparison & first & second & $p$ (Holm) & block bootstrap \\\\', '\\midrule']
for name, a, b, hk, ck in rows:
    L.append(f"{name} & {a} & {b} & {p(H['raw'][hk])} ({p(H['adjusted'][hk])}) & {bs(ck)} \\\\")
L += ['\\bottomrule', '\\end{tabular}']
open(OUT, 'w').write('% 自动生成：analysis/gen_table.py（勿手改）\n' + '\n'.join(L) + '\n'); print('\n'.join(L))
