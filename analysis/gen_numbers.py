"""从统计文件生成论文用的数字宏 paper/numbers.tex（会随数据更新的数一律走这里，不在正文手抄；10-07 第四位评审查出手抄数与 json 漂移）。
先跑 paper_stats.py 与 corrupt_structure.py，再跑本脚本，再编译。
用法: python3 gen_numbers.py logs numbers.tex"""
import sys, json

ROOT, OUT = sys.argv[1], sys.argv[2]
S = json.load(open(f'{ROOT}/paper_stats_v3.json'))
C = json.load(open(f'{ROOT}/corrupt_structure.json'))
M = {}
frac = lambda s: tuple(int(x) for x in s.split('/'))

# 发热（调用开始时的状态；单调用 fp32 暗屏）
lo = S['thermal_at_start_below_serious']; hi = S['thermal_at_start_serious_or_critical']
M['thLowWrong'], M['thLowN'] = lo['wrong'], lo['n']; M['thLowCIhi'] = lo['ci'][1]; M['thLowCIlo'] = lo['ci'][0]
M['thHighWrong'], M['thHighN'] = hi['wrong'], hi['n']; M['thHighPct'] = round(100 * hi['wrong'] / hi['n'])
M['thTotalN'] = lo['n'] + hi['n']; M['thLowRuns'] = S['thermal_at_start_low_runs']; M['thLowPhones'] = len(S['thermal_at_start_low_phones'])
ts = S['thermal_at_start_strata']
for key, name in [('A_start2', 'thAser'), ('A_start3', 'thAcrit'), ('B_start2', 'thBser'), ('C_start2', 'thCser')]:
    w, n = frac(ts[key]); M[name + 'W'], M[name + 'N'] = w, n
eb = S['thermal_below_serious']; M['thEndLowWrong'], M['thEndLowN'] = eb['wrong'], eb['n']
w, n = frac(S['battery_within_serious']['A_ge80']); M['batAhiW'], M['batAhiN'] = w, n
# 截断
W = C['整块']
M['trN'] = W['n']; M['trTail'] = W['tail_to_end']; M['trSufMed'] = round(100 * W['suffix_bad_frac_median'], 1)
M['trSufPten'] = round(100 * W['suffix_bad_frac_p10'], 1); M['trAfterFive'] = W['n_first_after_5pct']; M['trFirstMax'] = round(100 * W['first_pos_max'])
b16 = S['block16_cuts']
M['bsNZ'], M['bsBoundary'], M['bsFirstTok'], M['bsSixtyFour'] = b16['nonzero_wrong'], b16['at_chunk_boundary'], b16['at_first_token'], b16['at_multiple_of_64_not_chunk']
# 末尾校验（真机）
T = S['tailcheck_phone']
M['tcCaught'] = sum(v['tail_caught'] for v in T.values()); M['tcWrong'] = sum(v['wrong'] for v in T.values())
M['tcFA'] = sum(v['tail_false_alarm'] for v in T.values()); M['tcOK'] = sum(v['n'] - v['wrong'] for v in T.values())
M['tcCalls'] = sum(v['n'] for v in T.values())
M['tcOK'] = M['tcCalls'] - M['tcWrong']   # 正确输出数（摘要「441 次正确输出无一误报」）
M['tcNZ'] = sum(v['nz_wrong'] for v in T.values()); M['tcNZcaught'] = sum(v['nz_caught'] for v in T.values())
import math; M['tcFAhi'] = math.ceil(1000 * (1 - 0.025 ** (1 / M['tcOK']))) / 10   # 0 次误报时 Clopper-Pearson 95% 上限, 向上取到 0.1%   # 0 次误报时 Clopper-Pearson 95% 上限
# 发热门控三台
gA_u, gA_g = S['thermalgate_ungated'], S['thermalgate_gated']
M['gtAungW'], M['gtAgatW'] = gA_u['wrong'], gA_g['wrong']
for d in 'BC':
    M[f'gt{d}ungW'], M[f'gt{d}gatW'] = S[f'thermalgate_{d}_ungated']['wrong'], S[f'thermalgate_{d}_gated']['wrong']
M['gtAllungW'] = gA_u['wrong'] + S['thermalgate_B_ungated']['wrong'] + S['thermalgate_C_ungated']['wrong']
M['gtAllgatW'] = gA_g['wrong'] + S['thermalgate_B_gated']['wrong'] + S['thermalgate_C_gated']['wrong']
M['gtAllN'] = gA_u['n'] + S['thermalgate_B_ungated']['n'] + S['thermalgate_C_ungated']['n']
# 重启后
PR = S['postreboot_A']; M['prW'], M['prN'] = PR['wrong'], PR['n']; M['prII'] = PR['table']['II_wrong'] + PR['table']['II_ok']
M['prWiredLo'], M['prWiredHi'] = round(PR['wired_mb'][0] / 1000, 1), round(PR['wired_mb'][1] / 1000, 1)
P2 = S['postreboot2_A']; M['prTwoW'], M['prTwoN'] = P2['wrong'], P2['n']
PK = [k for k in S if k.startswith('prod_')]; M['prodW'] = sum(S[k]['wrong'] for k in PK); M['prodN'] = sum(S[k]['n'] for k in PK)
M['clipPooled'] = f"{1 - (1 - M['prodW'] / M['prodN']) ** 90:.1f}"
M['prodAtwoN'], M['prodBtwoN'] = S['prod_4s_phoneA2']['n'], S['prod_4s_phoneB2']['n']; M['prodAtwoW'], M['prodBtwoW'] = S['prod_4s_phoneA2']['wrong'], S['prod_4s_phoneB2']['wrong']
TO = json.load(open(f'{ROOT}/tail_overhead.json')); M['tcCostLo'] = round(100 * min(v['ratio_med'] for v in TO.values())); M['tcCostHi'] = round(100 * max(v['ratio_med'] for v in TO.values()))
M['tcCostLdLo'] = round(100 * min(v['ratio_with_load_med'] for v in TO.values())); M['tcCostLdHi'] = round(100 * max(v['ratio_with_load_med'] for v in TO.values()))
M['tcZero'] = M['tcWrong'] - M['tcNZ']
RS = S['reboot_series_A']; M['rbPreW'], M['rbPreN'] = RS['pre']['wrong'], RS['pre']['n']
for _x, _nm in zip(RS['post'], ['One', 'Two', 'Three']):
    M[f'rb{_nm}W'], M[f'rb{_nm}N'] = _x['wrong'], _x['n']; M[f'rb{_nm}Up'] = f"{_x['uptime_h'][0]}--{_x['uptime_h'][1]}"
M['rbTrendP'] = RS['trend_p_two_sided']
KL = S['klein']; M['klN'], M['klW'], M['klHot'], M['klLog'], M['klDt'], M['klCIhi'] = KL['klein_B']['n'], KL['klein_B']['wrong'], KL['klein_B']['hot'], KL['klein_B']['log_calls'], KL['klein_B']['dt_med'], KL['klein_B']['ci'][1]
M['klWanW'], M['klWanN'], M['klWanLog'] = KL['wan_after_B']['wrong'], KL['wan_after_B']['n'], KL['wan_after_B']['log_calls']
M['klWanAW'], M['klWanAN'] = KL['wan_after_A']['wrong'], KL['wan_after_A']['n']
KT = S['klein_total']; M['klTotN'], M['klTotW'], M['klTotCIhi'], M['klTotLog'] = KT['n'], KT['wrong'], KT['ci'][1], KT['log_calls']
M['klDtLo'] = min(KL['klein_A']['dt_med'], KL['klein_B']['dt_med']); M['klDtHi'] = max(KL['klein_A']['dt_med'], KL['klein_B']['dt_med'])
AL = json.load(open(f'{ROOT}/abort_log_link.json')); M['alCR'], M['alCN'] = AL['corrupted_with_record'], AL['corrupted_with_record'] + AL['corrupted_without_record']
M['alOR'], M['alON'] = AL['correct_with_record'], AL['correct_with_record'] + AL['correct_without_record']; M['alTot'] = M['alCN'] + M['alON']
TE = json.load(open(f'{ROOT}/tail_e2e.json'))['phoneA_s101_tail']; M['teChecks'], M['teFlag'], M['teZero'] = TE['tail_checks'], TE['flagged'], TE['zero_reruns']
M['teFlagMin'], M['teOkMax'] = TE['flagged_min'], f"{TE['ok_max']:.1e}"; M['teCost'] = round(100 * (TE['tail_load_s'] + TE['tail_run_s']) / TE['block_infer_s']); M['teMin'] = round(TE['clip_s'] / 60)
NC = S['newchip_A20']; M['ncN'], M['ncW'], M['ncHot'], M['ncLog'], M['ncCIhi'] = NC['gpu_n'], NC['gpu_wrong'], NC['gpu_hot'], NC['gpu_log'], NC['gpu_ci'][1]
M['ncDtLo'] = min(v['dt_med'] for k, v in NC['runs'].items() if k != 'cpu'); M['ncDtHi'] = max(v['dt_max'] for k, v in NC['runs'].items() if k != 'cpu'); M['ncCpuN'], M['ncCpuW'] = NC['runs']['cpu']['n'], NC['runs']['cpu']['wrong']
KC = S['klein_chain_A14']; M['kcN'], M['kcW'], M['kcHot'], M['kcLog'], M['kcDt'], M['kcCIhi'] = KC['klein_chain']['n'], KC['klein_chain']['wrong'], KC['klein_chain']['hot'], KC['klein_chain']['log_calls'], KC['klein_chain']['dt_med'], KC['klein_chain']['ci'][1]
M['kcWanW'], M['kcWanN'], M['kcWanDt'] = KC['wan_after']['wrong'], KC['wan_after']['n'], KC['wan_after']['dt_med']
SG = S['split_abort_segments']; M['sgA'], M['sgB'], M['sgC'] = SG.get('0', 0), SG.get('1', 0), SG.get('2', 0)
NN = S['newchip_chain3']; M['nnN'], M['nnW'], M['nnTen'], M['nnMax'], M['nnCIhi'], M['nnHot'] = NN['n'], NN['wrong'], NN['ge10s'], NN['dt_max'], NN['ci'][1], NN['hot']
M['tcSepMin'] = S['tailcheck_sep']['wrong_min']; M['thrCalls'] = S['threshold_check']['calls']
M['thrOkMax'] = f"{S['threshold_check']['ok_max']:.1e}"; M['thrWrongMin'] = f"{S['threshold_check']['wrong_min']:.1e}"
M['thHighPctExact'] = round(100 * hi['wrong'] / hi['n'], 1)
CLs = S['thermal_at_start_cluster']; M['thClRuns'] = CLs['runs']; M['thClLowHi'] = CLs['low_ci'][1]
M['thClDiffLo'], M['thClDiffHi'] = CLs['diff_ci']; M['thClLowRunsW'], M['thClLowRuns'] = CLs['low_runs_with_wrong'], CLs['low_runs']
# 其他
M['durAUC'] = f"{S['duration_auc_phoneA']:.2f}"
RA = S['recovery_phoneA']
for pol, nm in [('imm', 'Imm'), ('reload', 'Reload'), ('wait', 'Wait')]:
    M[f'recA{nm}R'], M[f'recA{nm}N'] = RA[pol]['recovered'], RA[pol]['needed']

with open(OUT, 'w') as f:
    f.write('% 自动生成：analysis/gen_numbers.py（勿手改）\n')
    for k, v in M.items(): f.write(f'\\newcommand{{\\n{k}}}{{{v}}}\n')
print(len(M), '个宏 →', OUT)
for k, v in M.items(): print(f'  \\n{k} = {v}')
