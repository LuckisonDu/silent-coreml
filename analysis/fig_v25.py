"""ICCE v25（6 页版）的三张率图：只读 paper_stats_v3.json，不手填数字。
fig/rates_char.pdf   —— 哪些条件下会静默出错（fp32 三台 / 30 Hz 界面 / 上线档 / 576 / 第 16 块 / Klein）
fig/rates_mitig.pdf  —— 防护：发热门控三台、拆段、段间停顿与保温
fig/reboot.pdf       —— 同一台（A）同一块：重启前 vs 重启后随开机时长
用法: python3 fig_v25.py logs <输出目录>"""
import sys, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT, OUT = sys.argv[1], sys.argv[2]
S = json.load(open(f'{ROOT}/paper_stats_v3.json'))
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': 0.6})

def cp(k, n):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return 100 * lo, 100 * hi

def wn(key):
    d = S
    for p in key.split('.'): d = d[p]
    return d['wrong'], d['n']

def forest(groups, fname, h):
    labels, vals, lo, hi, cols, ypos, hdr_pos, hdr_lab = [], [], [], [], [], [], [], []
    y = 0.0
    for gname, items, col in groups:
        hdr_pos.append(y); hdr_lab.append(gname); y += 0.75
        for lab, key in items:
            w, n = wn(key); v = 100 * w / n; a, b = cp(w, n)
            ypos.append(y); labels.append(f'{lab}  ({w}/{n})'); vals.append(v); lo.append(v - a); hi.append(b - v); cols.append(col); y += 0.75
        y += 0.35
    fig, ax = plt.subplots(figsize=(3.45, h))
    ax.barh(ypos, vals, xerr=[lo, hi], color=cols, height=0.55, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(ypos + hdr_pos); ax.set_yticklabels(labels + hdr_lab)
    for t in ax.get_yticklabels()[len(labels):]: t.set_fontweight('bold')
    ax.invert_yaxis(); ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 100); ax.set_xlabel('% corrupted (95% CI)')
    ax.grid(axis='x', lw=0.3, alpha=0.5); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.3); fig.savefig(f'{OUT}/{fname}.pdf'); fig.savefig(f'{OUT}/{fname}.png', dpi=200); plt.close(fig)

R, G, B, P = '#c0392b', '#7f8c8d', '#2471a3', '#8e44ad'
forest([
    ('Wan fp32 block 0, main set (19 s)', [('Phone A', 'fp32_phoneA'), ('Phone B', 'fp32_phoneB'), ('Phone C', 'fp32_phoneC')], R),
    ('UI refreshing at 30 Hz, Phone A', [('run 1', 'ui_phoneA_a'), ('run 2', 'ui_phoneA_b')], R),
    ('Wan shipped mixed precision (13-18 s)', [('4-s, A hot', 'prod_4s_phoneA'), ('4-s, A hot, later', 'prod_4s_phoneA2'), ('4-s, B', 'prod_4s_phoneB'),
                                               ('4-s, B hot', 'prod_4s_phoneB2'), ('5-s, B', 'prod_5s_phoneB'), ('5-s, C', 'prod_5s_phoneC')], G),
    ('Shorter work', [('Wan 576 px (11 s), B hot', 'ctl576_phoneB'), ('Klein block (3.3 s), A+B hot', 'klein_total')], G),
], 'rates_char', 2.85)
forest([
    ('Thermal gating (wait until below serious)', [('A ungated', 'thermalgate_ungated'), ('A gated', 'thermalgate_gated'), ('B ungated', 'thermalgate_B_ungated'),
                                                    ('B gated', 'thermalgate_B_gated'), ('C ungated', 'thermalgate_C_ungated'), ('C gated', 'thermalgate_C_gated')], B),
    ('Splitting block 0 into 3 calls, Phone A', [('one call', 'split_phoneA_whole'), ('3 calls', 'split_phoneA_3seg')], B),
    ('Idle pause between split calls, Phone B', [('one call', 'gap_phoneB_whole'), ('3 calls, 0 s', 'gap_phoneB_3seg_gap0'),
                                                 ('3 calls, 1 s', 'gap_phoneB_3seg_gap1'), ('3 calls, 3 s', 'gap_phoneB_3seg_gap3')], P),
    ('Pause vs. keeping the GPU busy, Phone A', [('one call', 'keepwarm_A_whole'), ('3 calls, 0 s', 'keepwarm_A_split_gap0'),
                                                 ('3 calls, 3 s idle', 'keepwarm_A_split_gap3_idle'), ('3 calls, 3 s busy', 'keepwarm_A_split_gap3_warm')], P),
], 'rates_mitig', 3.1)
# 重启图
RS = S['reboot_series_A']
xs, ys, lo, hi, lab = [], [], [], [], []
w, n = RS['pre']['wrong'], RS['pre']['n']; a, b = cp(w, n)
xs.append(-0.6); ys.append(100 * w / n); lo.append(100 * w / n - a); hi.append(b - 100 * w / n); lab.append(f'before\nrestart\n{w}/{n}')
for p in RS['post']:
    w, n = p['wrong'], p['n']; a, b = cp(w, n); x = sum(p['uptime_h']) / 2
    xs.append(x); ys.append(100 * w / n); lo.append(100 * w / n - a); hi.append(b - 100 * w / n); lab.append(f"{p['uptime_h'][0]}-{p['uptime_h'][1]} h\n{w}/{n}")
fig, ax = plt.subplots(figsize=(3.45, 1.45))
ax.errorbar(xs[1:], ys[1:], yerr=[lo[1:], hi[1:]], fmt='o-', color=R, lw=0.8, ms=3, capsize=2)
ax.errorbar(xs[:1], ys[:1], yerr=[lo[:1], hi[:1]], fmt='s', color='black', ms=3, capsize=2)
for i, (x, y, t) in enumerate(zip(xs, ys, lab)):
    off = (6, -2) if i == 0 else [(0, 11), (0, 17), (0, 12)][i - 1]
    ax.annotate(t.replace('\n', ' ', 1) if i else t, (x, y), textcoords='offset points', xytext=off, fontsize=5.5, va='center', ha='left' if i == 0 else 'center')
ax.axvline(-0.2, ls=':', lw=0.6, color='grey'); ax.set_xlim(-1, 5.6); ax.set_ylim(0, 100)
ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_xlabel('uptime after restart (h)'); ax.set_ylabel('corrupted (%)')
ax.text(0.98, 0.95, f"trend p = {RS['trend_p_two_sided']}", transform=ax.transAxes, ha='right', va='top', fontsize=6)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(f'{OUT}/reboot.pdf'); fig.savefig(f'{OUT}/reboot.png', dpi=200)
print('ok')
