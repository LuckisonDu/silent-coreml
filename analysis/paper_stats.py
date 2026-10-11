"""ICCE 静默失败论文 v3 的全部数字：从原始探针日志逐轮复算，写 paper_stats_v3.json（论文里的数只从这里取）。
每一组 = (目录, 该轮「开始」行的 UTC 时间前缀)；手机日志是累加的、同一行会出现在多个目录，所以按轮取而不是按目录全取。
判错：与参考 1−cos > 1e-3（正确输出 1−cos ≤ 3.4e-6）。日志记录：II = 影响交互(ImpactingInteractivity)，IV = 只有连带丢弃(InnocentVictim)。
置信区间：Clopper–Pearson 95%（假设各次独立；错误成串，区间偏乐观）。比较：Fisher 精确检验（双侧）。
用法: python3 paper_stats.py logs"""
import re, sys, json, math, statistics as st
from math import comb

ROOT = sys.argv[1]


def runs_in(dirname):
    L = open(f'{ROOT}/{dirname}/abort_probe.txt', errors='ignore').read().splitlines()
    starts = [i for i, l in enumerate(L) if 'abort-probe 开始 档=' in l]
    out = []
    for j, si in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(L)
        out.append((L[si], L[si:end], L[max(0, si - 5):si + 15]))   # (开始行, 本轮行, 找「额外停」用的窗口)
    return out


def get_run(dirname, ts_prefix, nth=0):
    """取 dirname 里「开始」行时间以 ts_prefix 开头的第 nth 轮"""
    hits = [r for r in runs_in(dirname) if r[0].startswith('[' + ts_prefix)]
    if len(hits) <= nth: raise SystemExit(f'找不到 {dirname} {ts_prefix}')
    head, lines, win = hits[nth]
    gap = next((re.search(r'额外停 ([\d.]+)', l).group(1) for l in win if '额外停' in l), '0')
    rows = []
    for l in lines:
        if 'abort-probe i=' not in l: continue
        g = lambda k, d=None: (re.search(k, l).group(1) if re.search(k, l) else d)
        m = re.search(r'影响交互 (\d+) / 连带丢弃 (\d+)', l)
        rec = 'none' if not m else ('II' if int(m.group(1)) > 0 else 'IV')
        oneminus = float(g(r'ref_1mcos=([^ ]+)', 'nan'))
        rows.append(dict(rec=rec, wrong=oneminus > 1e-3, threw=g(r'threw=(\S+)') != '-',
                         zero=float(g(r'zero=([\d.]+)', 0)), naninf=int(g(r'naninf=(\d+)', 0)),
                         dt=float(g(r'dt=([\d.]+)s')), th=int(g(r'th=(\d)')), bat=int(g(r'bat=(-?\d+)', -1)),
                         pwr=g(r'pwr=(\S+)'), tail=g(r'tailchk=(\S+)')))
    return dict(head=head[1:20], tier=re.search(r'档=(\S+)', head).group(1), gap=gap, rows=rows)


def cp(k, n, a=0.05):
    """Clopper–Pearson（二分求 Beta 分位），返回百分比 [lo, hi]"""
    def cdf(x, p):  # P(X<=x)
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))
    def solve(f, target):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            (lo, hi) = (mid, hi) if f(mid) > target else (lo, mid)
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else solve(lambda p: cdf(k - 1, p), 1 - a / 2)
    hi = 1.0 if k == n else solve(lambda p: cdf(k, p), a / 2)
    return [round(100 * lo, 1), round(100 * hi, 1)]


def fisher(a, n1, b, n2):
    """双侧 Fisher 精确检验：a/n1 vs b/n2"""
    K, N = a + b, n1 + n2
    p0 = comb(n1, a) * comb(n2, b) / comb(N, K)
    ps = [comb(n1, x) * comb(n2, K - x) / comb(N, K) for x in range(max(0, K - n2), min(K, n1) + 1)]
    return float('%.2g' % min(1.0, sum(p for p in ps if p <= p0 * (1 + 1e-9))))


def summarize(rows):
    n = len(rows); w = sum(r['wrong'] for r in rows)
    tab = {f'{rec}_{"wrong" if wr else "ok"}': sum(1 for r in rows if r['rec'] == rec and r['wrong'] == wr)
           for rec in ['II', 'IV', 'none'] for wr in [True, False]}
    nonzero_wrong = sum(1 for r in rows if r['wrong'] and r['zero'] < 0.99)
    usual_caught = sum(1 for r in rows if r['wrong'] and (r['threw'] or r['zero'] > 0.99 or r['naninf'] > 0))
    either = [r for r in rows if r['rec'] != 'none']
    return dict(n=n, wrong=w, wrong_pct=round(100 * w / max(n, 1), 1), wrong_ci=cp(w, n) if n else None,
                nonzero_wrong=nonzero_wrong, usual_checks_caught=usual_caught, table=tab,
                log_either_recall=f'{sum(r["wrong"] for r in either)}/{w}', log_either_false_alarm=f'{sum(not r["wrong"] for r in either)}/{n - w}',
                log_II_recall=f'{sum(r["wrong"] for r in rows if r["rec"] == "II")}/{w}', log_II_false_alarm=f'{sum(not r["wrong"] for r in rows if r["rec"] == "II")}/{n - w}',
                dt_median=round(st.median(r['dt'] for r in rows), 1) if rows else None,
                dt_range=[round(min(r['dt'] for r in rows), 1), round(max(r['dt'] for r in rows), 1)] if rows else None,
                thermal_serious=sum(r['th'] >= 2 for r in rows), battery_range=[min(r['bat'] for r in rows), max(r['bat'] for r in rows)] if rows else None)


S = {}
def add(key, dirname, ts, nth=0, take=None, skip=0, note=''):
    r = get_run(dirname, ts, nth); rows = r['rows'][skip:]
    if take: rows = rows[:take]
    S[key] = dict(source=f'{dirname} 「开始」{r["head"]} 档={r["tier"]} gap={r["gap"]}', note=note, **summarize(rows))
    return rows

# ① 静默错误：三台 iPhone 12、fp32 第 0 块、调暗静止
a = add('fp32_phoneA', 'probe2_1005_1357_phoneA', '2026-10-05 20:25', take=150, note='iOS 26.5.2，第 1 轮')
b = add('fp32_phoneB', 'probephoneB2_1005_1910', '2026-10-06 02:11', note='iOS 26.3.1')
c = add('fp32_phoneC', 'probephoneC_1005_2159', '2026-10-06 05:00', note='iOS 26.6.2')
S['fp32_three_phones'] = dict(note='三台合计', **summarize(a + b + c))
# ①b 界面每秒刷新 30 次（模拟看着进度条），phoneA fp32
add('ui_phoneA_a', 'probe3_1005_1656_phoneA', '2026-10-05 21:19', note='30 Hz 刷新，18 次后没电关机')
add('ui_phoneA_b', 'probe3_1005_1656_phoneA', '2026-10-06 00:03', note='30 Hz 刷新第二轮，20 次；2 次错而窗口内无记录')
# ② 现役档（产品配置）
add('prod_4s_phoneB', 'probephoneB3_1005_2126', '2026-10-06 04:28', note='704x1280m_t2，现役 4 秒档')
add('prod_5s_phoneB', 'probephoneB61_1006_0721', '2026-10-06 14:27', note='5 秒 704x1280mf61p（热机后）')
add('prod_4s_phoneA', 'probephoneAtail_1007_0003', '2026-10-07 08:12', note='704x1280m 现役 4 秒档，热（同轮 fp32 93% 错），带末尾校验')
add('prod_5s_phoneC', 'probephoneC61_1006_0721', '2026-10-06 14:30', note='5 秒，跑到 93 次停')
add('prod_4s_phoneA2', 'probephoneAshipped_1007_0753', '2026-10-07 15:00', note='实验 I：704x1280m 150 次全部热起步，带 mt64；手机重启后约 3 h')
add('prod_4s_phoneB2', 'probephoneBshipped_1007_0800', '2026-10-07 15:08', note='实验 I：704x1280m 150 次全部热起步，带 mt64')
# ③ 576 对照（同机热机后）
add('ctl576_phoneB', 'probephoneB576_1005_2157', '2026-10-06 05:04', note='576x1024 fp32，紧接 704 热机 20 次')
add('ctl576_phoneA', 'probephoneA4_1005_1948', '2026-10-06 03:30', note='凉机，20 次后被打断')
# ④ 预防：拆三段（phoneA，v2 之后两轮交替；不算热机）
w1 = get_run('probephoneA5split_1006_1106', '2026-10-06 18:15')['rows'] + get_run('probephoneA5split_1006_1106', '2026-10-06 18:36')['rows']
s1 = get_run('probephoneA5split_1006_1106', '2026-10-06 18:25')['rows'] + get_run('probephoneA5split_1006_1106', '2026-10-06 18:47')['rows']
S['split_phoneA_whole'] = dict(source='probephoneA5split_1006_1106 18:15+18:36', **summarize(w1))
S['split_phoneA_3seg'] = dict(source='probephoneA5split_1006_1106 18:25+18:47', **summarize(s1))
S['split_phoneA_fisher_p'] = fisher(sum(r['wrong'] for r in w1), len(w1), sum(r['wrong'] for r in s1), len(s1))
# ⑤ 段间空隙（phoneB 第一轮）+ 拆段 phoneB
wd = get_run('probephoneBgap_1006_1255', '2026-10-06 20:03')['rows'] + get_run('probephoneBgap_1006_1255', '2026-10-06 20:42')['rows']
S['gap_phoneB_whole'] = dict(source='probephoneBgap 20:03+20:42', **summarize(wd))
for ts, k in [('2026-10-06 20:11', 'gap0'), ('2026-10-06 20:21', 'gap1'), ('2026-10-06 20:31', 'gap3')]:
    rr = get_run('probephoneBgap_1006_1255', ts)['rows']; S[f'gap_phoneB_3seg_{k}'] = dict(source=f'probephoneBgap {ts}', **summarize(rr))
S['gap_phoneB_gap3_vs_gap0_fisher_p'] = fisher(S['gap_phoneB_3seg_gap3']['wrong'], 20, S['gap_phoneB_3seg_gap0']['wrong'], 20)
# ⑥ 注意力查询分块（phoneB 两轮交替）
for t, ts in [('q8', ['2026-10-06 18:57', '2026-10-06 19:26']), ('q16', ['2026-10-06 19:06', '2026-10-06 19:35']), ('q32', ['2026-10-06 19:15', '2026-10-06 19:44'])]:
    rr = sum((get_run('probephoneBq_1006_1149', x)['rows'] for x in ts), [])
    S[f'qchunk_phoneB_{t}'] = dict(source=f'probephoneBq_1006_1149 {ts}', **summarize(rr))
S['qchunk_q8_vs_q32_fisher_p'] = fisher(S['qchunk_phoneB_q8']['wrong'], 50, S['qchunk_phoneB_q32']['wrong'], 50)
S['qchunk_q8_vs_q16_fisher_p'] = fisher(S['qchunk_phoneB_q8']['wrong'], 50, S['qchunk_phoneB_q16']['wrong'], 50)
# ⑦ 恢复对比（phoneC，热机、错误多）
L = [l for l in open(f'{ROOT}/probephoneC_rec_1005_2258/abort_probe.txt', errors='ignore') if 'abort-recover ep=' in l]
rec = {}
for pol in ['imm', 'wait', 'reload']:
    v = [l for l in L if f'pol={pol} ' in l]
    att = [int(re.search(r'attempts=(\d+)', l).group(1)) for l in v]
    rec[pol] = dict(n=len(v), first_wrong=sum(re.search(r'seq=(\S+)', l).group(1).startswith('BAD') for l in v),
                    needed_rerun=sum(a > 1 for a in att), final_ok=sum('final_ok=1' in l for l in v),
                    still_flagged_after_8=sum('final_flag=1' in l for l in v),
                    rerun_total_s_median=round(st.median(float(re.search(r'total=([\d.]+)s', l).group(1)) for l, a in zip(v, att) if a > 1), 0))
S['recovery_phoneC'] = dict(source='probephoneC_rec_1005_2258 abort-recover 行', **rec)
S['recovery_phoneC_wait_vs_imm_fisher_p'] = fisher(rec['wait']['final_ok'], 30, rec['imm']['final_ok'], 30)
# ⑧ 端到端与错误结构
S['e2e'] = dict(tally=json.load(open(f'{ROOT}/e2e/e2e_tally.json')), compare=json.load(open(f'{ROOT}/e2e/e2e_compare.json')))
S['corruption_structure'] = json.load(open(f'{ROOT}/corrupt_structure.json'))
# ⑨ 发热 / 电量分层：所有 fp32 整块、暗屏、非分段的调用（按整行去重；th 为调用结束时的 thermalState）
import glob, collections
phone = lambda d: 'A' if ('phoneA' in d or d.startswith(('probe2', 'probe3', 'probephoneA4', 'probephoneA5', 'probe_'))) else ('C' if ('lk' in d or 'phoneC' in d) else 'B')
seen = set(); TH = collections.defaultdict(lambda: [0, 0]); BT = collections.defaultdict(lambda: [0, 0])
for f in sorted(glob.glob(f'{ROOT}/*/abort_probe.txt')):
    d = f.split('/')[-2]; tier = None; ui = False
    for l in open(f, errors='ignore'):
        if '开始 档=' in l: tier = re.search(r'档=(\S+)', l).group(1); ui = 'ui压力=true' in l; continue
        if 'abort-probe i=' not in l or l in seen: continue
        seen.add(l)
        if tier != '704x1280f32p' or ui or 'seg=' in l: continue
        w = float(re.search(r'ref_1mcos=([^ ]+)', l).group(1)) > 1e-3; th = int(re.search(r'th=(\d)', l).group(1))
        TH[f'{phone(d)}_th{th}'][0] += w; TH[f'{phone(d)}_th{th}'][1] += 1
        b = re.search(r'bat=(-?\d+)', l)
        if th >= 2 and b and int(b.group(1)) >= 0:
            bb = int(b.group(1)); k = f"{phone(d)}_{'lt45' if bb < 45 else ('45to80' if bb < 80 else 'ge80')}"
            BT[k][0] += w; BT[k][1] += 1
S['thermal_strata'] = {k: f'{v[0]}/{v[1]}' for k, v in sorted(TH.items())}
lowth = [sum(v[i] for k, v in TH.items() if not k.endswith('th2')) for i in (0, 1)]
S['thermal_below_serious'] = dict(wrong=lowth[0], n=lowth[1], ci=cp(lowth[0], lowth[1]))
S['battery_within_serious'] = {k: f'{v[0]}/{v[1]}' for k, v in sorted(BT.items())}
S['battery_A_ge80_vs_lt45_fisher_p'] = fisher(BT['A_ge80'][0], BT['A_ge80'][1], BT['A_lt45'][0], BT['A_lt45'][1])
# ⑩ 按「调用开始时」的发热状态分层：上一调用结束时的 th（同一轮内调用背靠背、间隔约 0.5 s），每轮第一次用「开始」行里的发热值
seen = set(); TS = collections.defaultdict(lambda: [0, 0]); LOWRUNS = set(); runhead = ''; CL = collections.defaultdict(list)
# 公开包: 目录已改名(机名→Phone A/B/C), 字母序变了; 下面的整轮自助法按目录顺序取轮, 故按原始目录顺序(logs/runs.tsv 的行序)遍历, 结果与论文逐位一致
import os as _os
_ORD = {l.split('\t')[0]: i for i, l in enumerate(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'logs', 'runs.tsv')).read().splitlines()[1:])}
for f in sorted(glob.glob(f'{ROOT}/*/abort_probe.txt'), key=lambda f: (_ORD.get(f.split('/')[-2], 1e9), f)):
    d = f.split('/')[-2]; tier = None; ui = False; prev = None
    for l in open(f, errors='ignore'):
        if '开始 档=' in l:
            tier = re.search(r'档=(\S+)', l).group(1); ui = 'ui压力=true' in l; runhead = l[1:20]
            m = re.search(r'发热 (\d)', l); prev = int(m.group(1)) if m else None; continue
        if 'abort-probe i=' not in l: continue
        th = int(re.search(r'th=(\d)', l).group(1))
        if l in seen: prev = th; continue
        seen.add(l)
        if tier == '704x1280f32p' and not ui and 'seg=' not in l and prev is not None:
            w = float(re.search(r'ref_1mcos=([^ ]+)', l).group(1)) > 1e-3
            k = f"{phone(d)}_start{prev}"; TS[k][0] += w; TS[k][1] += 1
            CL[(phone(d), runhead)].append((prev >= 2, w))
            if prev < 2: LOWRUNS.add((phone(d), runhead))
        prev = th
S['thermal_at_start_strata'] = {k: f'{v[0]}/{v[1]}' for k, v in sorted(TS.items())}
S['thermal_at_start_low_runs'] = len(LOWRUNS); S['thermal_at_start_low_phones'] = sorted({r[0] for r in LOWRUNS})
lo = [sum(v[i] for k, v in TS.items() if k[-1] in '01') for i in (0, 1)]; hi = [sum(v[i] for k, v in TS.items() if k[-1] in '23') for i in (0, 1)]
S['thermal_at_start_below_serious'] = dict(wrong=lo[0], n=lo[1], ci=cp(lo[0], lo[1]))
S['thermal_at_start_serious_or_critical'] = dict(wrong=hi[0], n=hi[1], pct=round(100 * hi[0] / hi[1], 1))
# ⑩b 簇稳健（10-07 第五位评审）：按轮整轮重抽样（轮 = 一次启动的一串调用），各层错误率的 95% 区间；低层另报「出过错的轮数 / 有低层调用的轮数」
import random as _r
_rnd = _r.Random(0); _keys = list(CL); _B = 4000; _lo, _hi, _d = [], [], []
for _ in range(_B):
    pick = [CL[_keys[_rnd.randrange(len(_keys))]] for _ in _keys]
    L = [w for c in pick for h, w in c if not h]; H = [w for c in pick for h, w in c if h]
    if L and H: _lo.append(sum(L) / len(L)); _hi.append(sum(H) / len(H)); _d.append(_hi[-1] - _lo[-1])
q = lambda v, a: sorted(v)[int(a * len(v))]
S['thermal_at_start_cluster'] = dict(runs=len(_keys), low_ci=[round(100 * q(_lo, .025), 2), round(100 * q(_lo, .975), 2)],
    high_ci=[round(100 * q(_hi, .025), 1), round(100 * q(_hi, .975), 1)], diff_ci=[round(100 * q(_d, .025), 1), round(100 * q(_d, .975), 1)],
    low_runs_with_wrong=sum(any(w for h, w in CL[k] if not h) for k in _keys), low_runs=sum(any(not h for h, w in CL[k]) for k in _keys))
# ⑩c 第二次重启后观察（实验 H 对照臂：不压舱，phoneA fp32 整块，三轮 25 次；10-07 05:16 panic 重启后约 2 h）
_h = [r for ts in ['2026-10-07 14:09', '2026-10-07 14:20', '2026-10-07 14:33'] for r in get_run('probephoneAballast_1007_0732', ts)['rows']]
S['postreboot2_A'] = dict(wrong=sum(int(r['wrong']) for r in _h), n=len(_h))
# ⑩d 同一台（A）同一块（第 0 块 fp32）重启前后与开机时长（10-07：重启 = 05:16 PDT 内核 panic；三段重启后都属同一次开机）
from scipy.stats import norm as _norm
import numpy as np
_pre = get_run('probephoneAtail_1007_0003', '2026-10-07 07:13')['rows']
_post = [('G', 'probephoneAuptime_1007_0620', ['2026-10-07 13:27']), ('H', 'probephoneAballast_1007_0732', ['2026-10-07 14:09', '2026-10-07 14:20', '2026-10-07 14:33']),
         ('G2', 'probephoneAuptime_1007_0931', ['2026-10-07 16:37'])]
_ser = []
for nm, d, tss in _post:
    rr = [r for ts in tss for r in get_run(d, ts)['rows']]
    up = []
    for ts in tss:
        _L = open(f'{ROOT}/{d}/abort_probe.txt', errors='ignore').read().splitlines(); _k = [i for i, l in enumerate(_L) if '开始' in l and l.startswith('[' + ts)][0]
        for l in _L[_k + 1:]:
            if '开始 档=' in l: break
            m = re.search(r'abort-probe i=.* uptime=([\d.]+)', l)
            if m: up.append(float(m.group(1)) / 3600)
    _ser.append(dict(name=nm, wrong=sum(int(r['wrong']) for r in rr), n=len(rr), uptime_h=[round(min(up), 1), round(max(up), 1)] if up else None))
_w = np.array([x['wrong'] for x in _ser]); _n = np.array([x['n'] for x in _ser]); _x = np.array([sum(x['uptime_h']) / 2 if x['uptime_h'] else i for i, x in enumerate(_ser)])
_p = _w.sum() / _n.sum(); _T = (_x * (_w - _n * _p)).sum(); _V = _p * (1 - _p) * ((_n * _x * _x).sum() - (_n * _x).sum() ** 2 / _n.sum())
S['reboot_series_A'] = dict(pre=dict(wrong=sum(int(r['wrong']) for r in _pre), n=len(_pre)), post=_ser, trend_p_two_sided=round(float(2 * (1 - _norm.cdf(abs(_T / np.sqrt(_V))))), 3),
                            pre_vs_post_fisher_p=fisher(sum(int(r['wrong']) for r in _pre), len(_pre), int(_w.sum()), int(_n.sum())))
# ⑩e 第二个模型族 Klein（FLUX.2 上线 704 三槽档 double_00，通用探针；实验 J）+ 前后 Wan fp32 阳性对照（同机）
def _gen_run(d, ts):
    _L = open(f'{ROOT}/{d}/abort_probe.txt', errors='ignore').read().splitlines(); _k = [i for i, l in enumerate(_L) if '开始 档=' in l and l.startswith('[' + ts)][0]; out = []
    for l in _L[_k + 1:]:
        if '开始 档=' in l: break
        if 'abort-probe i=' in l and 'ref_1mcos=' in l:
            out.append(dict(wrong=float(re.search(r'ref_1mcos=([^ ]+)', l).group(1)) > 1e-3, th0=int(re.search(r'th0=(\d)', l).group(1)), dt=float(re.search(r' dt=([\d.]+)', l).group(1)),
                            log=re.search(r'oslog=(\S+)', l).group(1) != '-'))
    return out
_K = {}
for nm, d, ts in [('klein_B', 'probephoneBklein_1007_0929', '2026-10-07 16:38'), ('wan_after_B', 'probephoneBklein_1007_0929', '2026-10-07 17:06'), ('klein_A', 'probephoneAklein_1007_1050', '2026-10-07 17:56'), ('wan_after_A', 'probephoneAklein_1007_1050', '2026-10-07 18:20')]:
    rr = _gen_run(d, ts); _K[nm] = dict(n=len(rr), wrong=sum(r['wrong'] for r in rr), hot=sum(r['th0'] >= 2 for r in rr), log_calls=sum(r['log'] for r in rr), dt_med=round(st.median(r['dt'] for r in rr), 2), ci=cp(sum(r['wrong'] for r in rr), len(rr)))
S['klein'] = _K
_kn = _K['klein_A']['n'] + _K['klein_B']['n']; _kw = _K['klein_A']['wrong'] + _K['klein_B']['wrong']; S['klein_total'] = dict(n=_kn, wrong=_kw, ci=cp(_kw, _kn), log_calls=_K['klein_A']['log_calls'] + _K['klein_B']['log_calls'])
# ⑩f 新芯片（iPhone 18 Pro / A20 / iOS 27.0.1，实验 M，10-08）
_NC = {}
for nm, ts in [('warm', '2026-10-08 07:11'), ('tail', '2026-10-08 07:13'), ('ui30', '2026-10-08 07:28'), ('cpu', '2026-10-08 07:35'), ('after', '2026-10-08 07:38')]:
    rr = _gen_run('probeiphone18pronewchip_1008_0010', ts)
    _NC[nm] = dict(n=len(rr), wrong=sum(r['wrong'] for r in rr), hot=sum(r['th0'] >= 2 for r in rr), log_calls=sum(r['log'] for r in rr),
                   dt_med=round(st.median(r['dt'] for r in rr), 2), dt_max=round(max(r['dt'] for r in rr), 2))
_g = [v for k, v in _NC.items() if k != 'cpu']
S['newchip_A20'] = dict(runs=_NC, gpu_n=sum(v['n'] for v in _g), gpu_wrong=sum(v['wrong'] for v in _g), gpu_hot=sum(v['hot'] for v in _g),
                        gpu_log=sum(v['log_calls'] for v in _g), gpu_ci=cp(sum(v['wrong'] for v in _g), sum(v['n'] for v in _g)))
# ⑩g 实验 P：另一模型家族的长调用（Klein 5 双流块链，A14 phoneA，10-08）+ 同机随后 Wan fp32 阳性对照
_KP = {}
for nm, ts in [('klein_chain', '2026-10-08 14:56'), ('wan_after', '2026-10-08 15:14')]:
    rr = _gen_run('probephoneAkleinchain_1008_0747', ts)
    _KP[nm] = dict(n=len(rr), wrong=sum(r['wrong'] for r in rr), hot=sum(r['th0'] >= 2 for r in rr), log_calls=sum(r['log'] for r in rr),
                   dt_med=round(st.median(r['dt'] for r in rr), 1), dt_max=round(max(r['dt'] for r in rr), 1), ci=cp(sum(r['wrong'] for r in rr), len(rr)))
S['klein_chain_A14'] = _KP
# 拆段 Wan 的中止记录落在哪一段（去重后按事后回填的 postseg 行）
import glob as _gl2, collections as _co2
_seen = set(); _seg = _co2.Counter()
for _f in _gl2.glob(f'{ROOT}/*/abort_probe.txt'):
    for _l in open(_f, errors='ignore'):
        _l = _l.strip()
        if _l in _seen: continue
        _seen.add(_l); _m = re.search(r'postseg i=\d+ seg=(\d) ii=(\d+) iv=(\d+)', _l)
        if _m: _seg[_m.group(1)] += 1
S['split_abort_segments'] = dict(_seg)
# ⑩h 实验 N：新芯片上的长调用（Wan 第 0–2 块连成一次调用，iphone18pro，10-08）
_NN = [r for ts in ['2026-10-08 13:59', '2026-10-08 14:01', '2026-10-08 14:10'] for r in _gen_run('probeiphone18prochain3_1008_0658', ts)]
S['newchip_chain3'] = dict(n=len(_NN), wrong=sum(r['wrong'] for r in _NN), hot=sum(r['th0'] >= 2 for r in _NN), ge10s=sum(r['dt'] >= 10 for r in _NN),
                           dt_max=round(max(r['dt'] for r in _NN), 1), log_calls=sum(r['log'] for r in _NN), ci=cp(sum(r['wrong'] for r in _NN), len(_NN)))
# ⑪ 发热门控（phoneA，交替两轮：不门控 / 门控每次调用前严重发热就等降温，上限 600 s；不算热机）
gt = {'ungated': ['2026-10-07 05:49', '2026-10-07 06:25'], 'gated': ['2026-10-07 05:59', '2026-10-07 06:35']}
for k, tss in gt.items():
    rr = sum((get_run('probephoneAthermal_1006_2240', t)['rows'] for t in tss), [])
    lines = [l for t in tss for l in open(f'{ROOT}/probephoneAthermal_1006_2240/abort_probe.txt', errors='ignore') if False]
    S[f'thermalgate_{k}'] = dict(source=f'probephoneAthermal_1006_2240 {tss}', **summarize(rr))
G = open(f'{ROOT}/probephoneAthermal_1006_2240/abort_probe.txt', errors='ignore').read()
waits = {}
for k, tss in gt.items():
    tot = 0.0; n = 0
    for t in tss:
        seg = G[G.index('[' + t):]; seg = seg[:seg.find('abort-probe 开始', 30) if seg.find('abort-probe 开始', 30) > 0 else len(seg)]
        for m in re.finditer(r'gate=(-?[\d.]+)', seg): tot += max(0.0, float(m.group(1))); n += 1
    waits[k] = dict(total_wait_s=round(tot), calls=n, wait_per_call_s=round(tot / max(n, 1), 1))
S['thermalgate_waits'] = waits
S['thermalgate_fisher_p'] = fisher(S['thermalgate_ungated']['wrong'], S['thermalgate_ungated']['n'], S['thermalgate_gated']['wrong'], S['thermalgate_gated']['n'])
# ⑫ 末尾校验真机（phoneA，fp32 + tail64 100 次；现役 704x1280m + mt64 100 次）。校验调用本身报错(threw)按「需重算」计
TC = {}
for tier, ts in [('fp32', '2026-10-07 07:13'), ('shipped4s', '2026-10-07 08:12')]:
    r = get_run('probephoneAtail_1007_0003', ts); rows = r['rows']
    w = [x for x in rows if x['wrong']]; ok = [x for x in rows if not x['wrong']]
    f = lambda x: x['tail'] is None or x['tail'].startswith('threw') or float(x['tail'].split('/')[0]) > 1e-4
    TC[tier] = dict(source=f"probephoneAtail_1007_0003 「开始」{r['head']} 档={r['tier']}", n=len(rows), wrong=len(w),
                    nz_wrong=sum(x['zero'] <= 0.99 for x in w), nz_caught=sum(f(x) for x in w if x['zero'] <= 0.99), tail_caught=sum(f(x) for x in w), tail_false_alarm=sum(f(x) for x in ok), verifier_threw=sum(1 for x in rows if x['tail'] and x['tail'].startswith('threw')),
                    tail_1mcos_ok_max=max(float(x['tail'].split('/')[0]) for x in ok if x['tail'] and not x['tail'].startswith('threw')) if ok else None,
                    tail_1mcos_wrong_min=min(float(x['tail'].split('/')[0]) for x in w if x['tail'] and not x['tail'].startswith('threw')) if w else None,
                    log_caught=sum(x['rec'] != 'none' for x in w), log_false_alarm=sum(x['rec'] != 'none' for x in ok), dt_median=round(st.median(x['dt'] for x in rows), 1))
r = get_run('probephoneAb16_1007_0337', '2026-10-07 10:48'); rows = r['rows']
w = [x for x in rows if x['wrong']]; ok = [x for x in rows if not x['wrong']]
f = lambda x: x['tail'] is None or x['tail'].startswith('threw') or float(x['tail'].split('/')[0]) > 1e-4
TC['block16_fp32'] = dict(source=f"probephoneAb16_1007_0337 「开始」{r['head']} 档={r['tier']}（block16，输入 blk15_out）", n=len(rows), wrong=len(w),
                          all_zero=sum(x['zero'] > 0.99 for x in w), nz_wrong=sum(x['zero'] <= 0.99 for x in w), nz_caught=sum(f(x) for x in w if x['zero'] <= 0.99), tail_caught=sum(f(x) for x in w), tail_false_alarm=sum(f(x) for x in ok),
                          log_caught=sum(x['rec'] != 'none' for x in w), log_false_alarm=sum(x['rec'] != 'none' for x in ok), dt_median=round(st.median(x['dt'] for x in rows), 1))
for dev, dd, ts in [('phoneB', 'probephoneBtailgate_1007_0506', '2026-10-07 12:14'), ('phoneC', 'probephoneCtailgate_1007_0500', '2026-10-07 12:09')]:
    r = get_run(dd, ts); rows = r['rows']; w = [x for x in rows if x['wrong']]; ok = [x for x in rows if not x['wrong']]
    TC[f'{dev}_fp32'] = dict(source=f"{dd} 「开始」{r['head']}", n=len(rows), wrong=len(w), nz_wrong=sum(x['zero'] <= 0.99 for x in w), nz_caught=sum(f(x) for x in w if x['zero'] <= 0.99), tail_caught=sum(f(x) for x in w), tail_false_alarm=sum(f(x) for x in ok),
                             log_caught=sum(x['rec'] != 'none' for x in w), log_false_alarm=sum(x['rec'] != 'none' for x in ok), dt_median=round(st.median(x['dt'] for x in rows), 1))
for dev, dd, ts in [('shippedA2', 'probephoneAshipped_1007_0753', '2026-10-07 15:00'), ('shippedB2', 'probephoneBshipped_1007_0800', '2026-10-07 15:08')]:
    r = get_run(dd, ts); rows = r['rows']; w = [x for x in rows if x['wrong']]; ok = [x for x in rows if not x['wrong']]
    TC[dev] = dict(source=f"{dd} 「开始」{r['head']}（实验 I 现役档）", n=len(rows), wrong=len(w), nz_wrong=sum(x['zero'] <= 0.99 for x in w), nz_caught=sum(f(x) for x in w if x['zero'] <= 0.99), tail_caught=sum(f(x) for x in w), tail_false_alarm=sum(f(x) for x in ok),
                   log_caught=sum(x['rec'] != 'none' for x in w), log_false_alarm=sum(x['rec'] != 'none' for x in ok), dt_median=round(st.median(x['dt'] for x in rows), 1))
for v in TC.values(): pass
S['tailcheck_phone'] = TC
# 末尾校验：所有真机轮次里错误输出的最小校验差、正确输出的最大校验差
_mins, _maxs = [], []
for dd, ts in [('probephoneAtail_1007_0003', '2026-10-07 07:13'), ('probephoneAtail_1007_0003', '2026-10-07 08:12'), ('probephoneAb16_1007_0337', '2026-10-07 10:48'),
               ('probephoneBtailgate_1007_0506', '2026-10-07 12:14'), ('probephoneCtailgate_1007_0500', '2026-10-07 12:09')]:
    for x in get_run(dd, ts)['rows']:
        if x['tail'] and not x['tail'].startswith('threw'):
            v = float(x['tail'].split('/')[0]); (_mins if x['wrong'] else _maxs).append(v)
S['tailcheck_sep'] = dict(wrong_min=round(min(_mins), 3), ok_max=max(_maxs))
# 全部探针调用的判错值分布（阈值不敏感的依据）
import glob as _g
_seen = set(); _v = []
for f in _g.glob(f'{ROOT}/*/abort_probe.txt'):
    for l in open(f, errors='ignore'):
        if 'abort-probe i=' not in l or l in _seen: continue
        _seen.add(l); m = re.search(r'ref_1mcos=([^ ]+)', l)
        try: _v.append(float(m.group(1)))
        except Exception: pass
S['threshold_check'] = dict(calls=len(_v), ok_max=max(x for x in _v if x <= 1e-3), wrong_min=min(x for x in _v if x > 1e-3), in_between=sum(3.4e-6 < x <= 1e-3 for x in _v))
# ⑪b 发热门控在 B、C 上的复现（交替两轮各 20 次）
for dev, dd, ung, gat in [('B', 'probephoneBtailgate_1007_0506', ['2026-10-07 12:46', '2026-10-07 13:16'], ['2026-10-07 12:53', '2026-10-07 13:22']),
                          ('C', 'probephoneCtailgate_1007_0500', ['2026-10-07 12:41', '2026-10-07 13:12'], ['2026-10-07 12:49', '2026-10-07 13:19'])]:
    for k, tss in [('ungated', ung), ('gated', gat)]:
        S[f'thermalgate_{dev}_{k}'] = dict(source=f'{dd} {tss}', **summarize(sum((get_run(dd, t)['rows'] for t in tss), [])))
    S[f'thermalgate_{dev}_fisher_p'] = fisher(S[f'thermalgate_{dev}_ungated']['wrong'], 40, S[f'thermalgate_{dev}_gated']['wrong'], 40)
# ⑰ 重启前后（phoneA 10-07 05:16 重启；JetsamEvent 05:09:50 时 wired 2.57 GB）
r = get_run('probephoneAuptime_1007_0620', '2026-10-07 13:27'); rows = r['rows']
Lu = open(f'{ROOT}/probephoneAuptime_1007_0620/abort_probe.txt', errors='ignore').read()
seg = Lu[Lu.index('[2026-10-07 13:27'):]
wired = [int(x) for x in re.findall(r'wired=(\d+)', seg)]; up = [float(x) for x in re.findall(r'uptime=(\d+)', seg)]
S['postreboot_A'] = dict(source='probephoneAuptime_1007_0620 13:27（重启后约 1.1–1.6 h）', **summarize(rows),
                         wired_mb=[min(wired), max(wired)], uptime_h=[round(min(up) / 3600, 1), round(max(up) / 3600, 1)])
# ⑬ 端到端处理组里被判废的块：查日志组 = 全零检查先抓的 + 日志额外判的
E2 = json.load(open(f'{ROOT}/e2e/e2e_tally.json'))
logarm = [x for x in E2 if x['arm'] == 'log' and 'phoneC' not in x['clip']]
S['e2e_log_arm_flagged'] = dict(all_zero=sum(x['zero_reruns'] for x in logarm), log_only=sum(x['log_reruns'] for x in logarm),
                                total=sum(x['zero_reruns'] + x['log_reruns'] for x in logarm), note='log_only 那 1 次重算后结果不变 = 误报')
# ⑭ 保温实验（phoneA，交替两轮：整块 / 三段不停 / 三段停 3 s 空闲 / 三段停 3 s 保温；不算热机）
WD = 'probephoneAwarm_1007_0159'
warm = {'whole': ['2026-10-07 09:08', '2026-10-07 09:51'], 'split_gap0': ['2026-10-07 09:18', '2026-10-07 10:01'],
        'split_gap3_idle': ['2026-10-07 09:28', '2026-10-07 10:11'], 'split_gap3_warm': ['2026-10-07 09:39', '2026-10-07 10:23']}
for k, tss in warm.items():
    S[f'keepwarm_A_{k}'] = dict(source=f'{WD} {tss}', **summarize(sum((get_run(WD, t)['rows'] for t in tss), [])))
kw = lambda k: (S[f'keepwarm_A_{k}']['wrong'], S[f'keepwarm_A_{k}']['n'])
S['keepwarm_A_fisher'] = dict(whole_vs_split=fisher(*kw('whole'), *kw('split_gap0')), idle_vs_gap0=fisher(*kw('split_gap3_idle'), *kw('split_gap0')),
                              warm_vs_idle=fisher(*kw('split_gap3_warm'), *kw('split_gap3_idle')), warm_vs_gap0=fisher(*kw('split_gap3_warm'), *kw('split_gap0')))
# ⑮ 评审要的出处补齐
# (a) 只看耗时的检测：主集 Phone A 150 次，错的调用返回更早 ⇒ 分数 = −dt；AUC = P(错的 dt < 对的 dt)
ra = get_run('probe2_1005_1357_phoneA', '2026-10-05 20:25')['rows'][:150]
wd = [x['dt'] for x in ra if x['wrong']]; od = [x['dt'] for x in ra if not x['wrong']]
S['duration_auc_phoneA'] = round(sum((a < b) + 0.5 * (a == b) for a in wd for b in od) / (len(wd) * len(od)), 3)
# (b) 恢复对比 Phone A（phoneA，probephoneA4）按「需要重算的回合」计
RA = [l for l in open(f'{ROOT}/probephoneA4_1005_1948/abort_probe.txt', errors='ignore') if 'abort-recover ep=' in l]
S['recovery_phoneA'] = {pol: dict(needed=sum(int(re.search(r'attempts=(\d+)', l).group(1)) > 1 for l in RA if f'pol={pol} ' in l),
                                  recovered=sum(int(re.search(r'attempts=(\d+)', l).group(1)) > 1 and 'final_ok=1' in l for l in RA if f'pol={pol} ' in l))
                        for pol in ['imm', 'wait', 'reload']}
# (c) 第 16 块非零错误的截断位置
L16 = open(f'{ROOT}/probephoneAb16_1007_0337/abort_probe.txt', errors='ignore').read().splitlines()
s16 = max(i for i, l in enumerate(L16) if '开始 档=704x1280f32b16' in l)
nz = []
for l in L16[s16:]:
    if 'abort-probe i=' not in l or float(re.search(r'ref_1mcos=([^ ]+)', l).group(1)) <= 1e-3: continue
    if float(re.search(r'zero=([\d.]+)', l).group(1)) > 0.99: continue
    nz.append((int(re.search(r'first=(-?\d+)', l).group(1)), int(re.search(r'last=(-?\d+)', l).group(1))))
S['block16_cuts'] = dict(nonzero_wrong=len(nz), to_last_token=sum(b == 11439 for a, b in nz), at_chunk_boundary=sum(a % 1430 == 0 for a, b in nz),
                         at_first_token=sum(a == 0 for a, b in nz), at_multiple_of_64_not_chunk=sum(a % 64 == 0 and a % 1430 != 0 for a, b in nz))
# ⑯ Holm 校正：论文报的 8 个主检验
tests = {'split_A': S['split_phoneA_fisher_p'], 'split_A_rep': S['keepwarm_A_fisher']['whole_vs_split'], 'gap_B': S['gap_phoneB_gap3_vs_gap0_fisher_p'],
         'gap_A': S['keepwarm_A_fisher']['idle_vs_gap0'], 'gating_A': S['thermalgate_fisher_p'], 'recovery_C': S['recovery_phoneC_wait_vs_imm_fisher_p'],
         'qchunk_B': S['qchunk_q8_vs_q32_fisher_p'], 'warm_vs_idle_A': S['keepwarm_A_fisher']['warm_vs_idle'],
         'gating_B': S['thermalgate_B_fisher_p'], 'gating_C': S['thermalgate_C_fisher_p'],
         'split_B': fisher(S['gap_phoneB_whole']['wrong'], S['gap_phoneB_whole']['n'], S['gap_phoneB_3seg_gap0']['wrong'], S['gap_phoneB_3seg_gap0']['n'])}
order = sorted(tests, key=tests.get); m = len(order); adj = {}; run = 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, (m - i) * tests[k])); adj[k] = float('%.2g' % run)
S['holm'] = dict(raw=tests, adjusted=adj)
json.dump(S, open(f'{ROOT}/paper_stats_v3.json', 'w'), ensure_ascii=False, indent=1)
for k, v in S.items():
    if isinstance(v, dict) and 'wrong_pct' in v:
        print(f"{k:32s} n={v['n']:3d} 错 {v['wrong']:3d} ({v['wrong_pct']}%, CI {v['wrong_ci']}) 常规检查抓到 {v['usual_checks_caught']} 日志两类 {v['log_either_recall']} 误报 {v['log_either_false_alarm']} dt {v['dt_median']}")
    elif not isinstance(v, dict): print(k, v)
print('recovery', rec)
