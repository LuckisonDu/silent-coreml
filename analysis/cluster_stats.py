"""簇稳健统计（ICCE v7 评审：错误成串出现，独立假设下的 p 偏乐观）。
对关键对比做「移动块自助法」(moving block bootstrap)：每组按原调用顺序取各轮的 0/1 序列，按长度 L 的连续块重抽样（块不跨轮），
重算两组错误率之差，取 2.5% / 97.5% 分位当 95% 区间；同时报「差 ≤ 0」的自助比例当单侧 p。L 取 5 与 10 两种。
用法: python3 cluster_stats.py logs"""
import sys, re, random, json
sys.argv = [sys.argv[0], sys.argv[1]]
ROOT = sys.argv[1]
exec(open(__file__.replace('cluster_stats.py', 'paper_stats.py')).read().split('S = {}')[0])


def seqs(dirname, ts_list):
    return [[int(r['wrong']) for r in get_run(dirname, ts)['rows']] for ts in ts_list]


def boot(A, B, L, n=4000, seed=0):
    rnd = random.Random(seed)
    def blocks(runs):
        bl = []
        for r in runs:
            for i in range(0, max(1, len(r) - L + 1)): bl.append(r[i:i + L])
        return bl
    ba, bb = blocks(A), blocks(B); na, nb = sum(map(len, A)), sum(map(len, B))
    def draw(bl, n_):
        out = []
        while len(out) < n_: out += rnd.choice(bl)
        return out[:n_]
    d = sorted(sum(draw(ba, na)) / na - sum(draw(bb, nb)) / nb for _ in range(n))
    obs = sum(map(sum, A)) / na - sum(map(sum, B)) / nb
    return dict(diff=round(100 * obs, 1), ci=[round(100 * d[int(0.025 * n)], 1), round(100 * d[int(0.975 * n)], 1)], p_le0=round(sum(x <= 0 for x in d) / n, 4))


R = {}
whole = seqs('probephoneA5split_1006_1106', ['2026-10-06 18:15', '2026-10-06 18:36'])
split = seqs('probephoneA5split_1006_1106', ['2026-10-06 18:25', '2026-10-06 18:47'])
g0 = seqs('probephoneBgap_1006_1255', ['2026-10-06 20:11']); g3 = seqs('probephoneBgap_1006_1255', ['2026-10-06 20:31'])
for L in (5, 10):
    R[f'split_A_whole_minus_3seg_L{L}'] = boot(whole, split, L)
    R[f'gap_B_3s_minus_0s_L{L}'] = boot(g3, g0, L)
gu = seqs('probephoneAthermal_1006_2240', ['2026-10-07 05:49', '2026-10-07 06:25']); gg = seqs('probephoneAthermal_1006_2240', ['2026-10-07 05:59', '2026-10-07 06:35'])
for L in (5, 10): R[f'thermalgate_A_ungated_minus_gated_L{L}'] = boot(gu, gg, L)
# ★v29（外审「多数比较只按调用独立」）：其余比较也做整段重抽样
def S2(d, ts): return seqs(d, ts)
more = {
    'thermalgate_B_ungated_minus_gated': (S2('probephoneBtailgate_1007_0506', ['2026-10-07 12:46', '2026-10-07 13:16']), S2('probephoneBtailgate_1007_0506', ['2026-10-07 12:53', '2026-10-07 13:22'])),
    'thermalgate_C_ungated_minus_gated': (S2('probephoneCtailgate_1007_0500', ['2026-10-07 12:41', '2026-10-07 13:12']), S2('probephoneCtailgate_1007_0500', ['2026-10-07 12:49', '2026-10-07 13:19'])),
    'split_A_hot_whole_minus_3seg': (S2('probephoneAwarm_1007_0159', ['2026-10-07 09:08', '2026-10-07 09:51']), S2('probephoneAwarm_1007_0159', ['2026-10-07 09:18', '2026-10-07 10:01'])),
    'split_B_whole_minus_3seg': (S2('probephoneBgap_1006_1255', ['2026-10-06 20:03', '2026-10-06 20:42']), S2('probephoneBgap_1006_1255', ['2026-10-06 20:11'])),
    'gap_A_3s_minus_0s': (S2('probephoneAwarm_1007_0159', ['2026-10-07 09:28', '2026-10-07 10:11']), S2('probephoneAwarm_1007_0159', ['2026-10-07 09:18', '2026-10-07 10:01'])),
    'warm_A_idle_minus_busy': (S2('probephoneAwarm_1007_0159', ['2026-10-07 09:28', '2026-10-07 10:11']), S2('probephoneAwarm_1007_0159', ['2026-10-07 09:39', '2026-10-07 10:23'])),
    'qchunk_B_q8_minus_q32': (S2('probephoneBq_1006_1149', ['2026-10-06 18:57', '2026-10-06 19:26']), S2('probephoneBq_1006_1149', ['2026-10-06 19:15', '2026-10-06 19:44'])),
}
for k, (A, B) in more.items(): R[f'{k}_L5'] = boot(A, B, 5)
json.dump(R, open(f'{ROOT}/cluster_stats.json', 'w'), indent=1)
for k, v in R.items(): print(k, v)
