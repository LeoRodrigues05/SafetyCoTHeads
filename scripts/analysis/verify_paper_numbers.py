#!/usr/bin/env python3
"""Re-derive every number asserted in the edited paper text and diff against it."""
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "common"))
import _bootstrap  # noqa: E402,F401  (puts src/ and every scripts/<group>/ on sys.path)
import json, sys
from statistics import mean
sys.path.insert(0, 'src')

rows = json.load(open('runs/direction_a_v6/reports/cell_metrics.json'))['rows']
ST = ['qwen3_8b', 'llama31_8b_control', 'olmo3_7b_think']
FAM = {}
for c in ['steering_a0.5', 'steering_a1.0', 'steering_a1.5']:
    FAM[c] = 'Steering'
FAM['steering_ablate'] = 'Directional ablation'
for c in ['ships_top3', 'ships_top5', 'ships_top8']:
    FAM[c] = 'SHIPS (heads)'
for c in ['neurons_top256', 'neurons_top512', 'neurons_top1024']:
    FAM[c] = 'Neuron'
prim = [r for r in rows if r['condition'] in FAM]


def av(sub, k):
    v = [r[k] for r in sub if isinstance(r.get(k), (int, float))]
    return mean(v) if v else None


checks = []


def chk(claim, got, want, tol=0.005):
    ok = got is not None and abs(got - want) <= tol
    checks.append((ok, claim, got, want))


# family table
for fam, P, Q, S, SFS in [('Steering', .32, .70, 1.00, .46),
                          ('Directional ablation', .07, 1.00, 1.00, .35),
                          ('SHIPS (heads)', .04, .96, 1.00, .24),
                          ('Neuron', .05, .95, 1.00, .19)]:
    sub = [r for r in prim if FAM[r['condition']] == fam and r['model'] in ST]
    chk(f'Tab1 {fam} P', av(sub, 'P'), P)
    chk(f'Tab1 {fam} Q', av(sub, 'Q'), Q)
    chk(f'Tab1 {fam} S', av(sub, 'S_v6'), S)
    chk(f'Tab1 {fam} SFS', av(sub, 'SFS'), SFS)

# per-model family SFS (heatmap + appendix table)
want = {'qwen3_8b': {'Directional ablation': .41, 'Neuron': .04, 'SHIPS (heads)': .07, 'Steering': .31},
        'llama31_8b_control': {'Directional ablation': .52, 'Neuron': .44, 'SHIPS (heads)': .45, 'Steering': .66},
        'olmo3_7b_think': {'Directional ablation': .14, 'Neuron': .09, 'SHIPS (heads)': .19, 'Steering': .38}}
for m, d in want.items():
    for fam, v in d.items():
        sub = [r for r in prim if r['model'] == m and FAM[r['condition']] == fam]
        chk(f'heatmap {m}/{fam}', av(sub, 'SFS'), v)

# Llama steering Q ladder + specific cells
for cond, q in [('steering_a0.5', .98), ('steering_a1.0', .41), ('steering_a1.5', .14)]:
    sub = [r for r in prim if r['model'] == 'llama31_8b_control' and r['condition'] == cond]
    chk(f'Llama {cond} macro Q', av(sub, 'Q'), q)
r = [x for x in rows if x['model'] == 'llama31_8b_control' and x['dataset'] == 'jbb' and x['condition'] == 'steering_a0.5'][0]
for k, v, lbl in [('P', .70, 'P'), ('Q', .98, 'Q'), ('S_v6', 1.00, 'S'), ('SFS', .88, 'SFS')]:
    chk(f'teaser Llama a0.5 jbb {lbl}', r[k], v)
r = [x for x in rows if x['model'] == 'llama31_8b_control' and x['dataset'] == 'jbb' and x['condition'] == 'steering_a1.5'][0]
for k, v, lbl in [('hac', .80, 'HAC'), ('Q', .05, 'Q'), ('SFS', .34, 'SFS')]:
    chk(f'teaser Llama a1.5 jbb {lbl}', r[k], v)
r = [x for x in rows if x['model'] == 'olmo3_7b_base' and x['dataset'] == 'jbb' and x['condition'] == 'neurons_top512'][0]
for k, v, lbl in [('hac', .55, 'HAC'), ('P', .00, 'P'), ('Q', 1.00, 'Q')]:
    chk(f'teaser Base neurons512 jbb {lbl}', r[k], v)

# OLMo-3-Think
sub = [r for r in prim if r['model'] == 'olmo3_7b_think' and r['condition'] == 'steering_a1.0']
chk('OLMo-Think a1.0 SFS', av(sub, 'SFS'), .51)
chk('OLMo-Think a1.0 P', av(sub, 'P'), .18)
sub = [r for r in prim if r['model'] == 'olmo3_7b_think' and r['condition'] == 'steering_a1.5']
chk('OLMo-Think a1.5 Q', av(sub, 'Q'), .00)

# Qwen strongest steering dose
sub = [r for r in prim if r['model'] == 'qwen3_8b' and r['condition'] == 'steering_a1.5']
chk('Qwen a1.5 SFS', av(sub, 'SFS'), .44)
qmin = min(r['Q'] for r in rows if r['model'] == 'qwen3_8b' and r['condition'].startswith('steering_a'))
checks.append((qmin >= 0.93, 'Qwen min steering Q >= 0.93', qmin, 0.93))

# grid averages
for m, v in [('qwen3_8b', .17), ('olmo3_7b_think', .19), ('llama31_8b_control', .52)]:
    sub = [r for r in prim if r['model'] == m]
    chk(f'grid-avg SFS {m}', av(sub, 'SFS'), v)

# appendix dose table
for cond, P, Q, S, SFS in [('steering_a0.5', .224, .991, .995, .424),
                           ('steering_a1.0', .355, .749, .998, .515),
                           ('steering_a1.5', .394, .365, .997, .433),
                           ('ships_top3', .046, .953, .997, .215),
                           ('ships_top5', .063, .988, .996, .271),
                           ('ships_top8', .025, .936, .998, .228),
                           ('neurons_top256', .021, .955, .998, .203),
                           ('neurons_top512', .017, .972, .996, .118),
                           ('neurons_top1024', .116, .915, .994, .251)]:
    sub = [r for r in prim if r['model'] in ST and r['condition'] == cond]
    for k, v, lbl in [('P', P, 'P'), ('Q', Q, 'Q'), ('S_v6', S, 'S'), ('SFS', SFS, 'SFS')]:
        chk(f'dose {cond} {lbl}', av(sub, k), v, tol=0.0015)

# transfer table
for fam, t, s in [('Steering', .07, .04), ('Directional ablation', .11, .08),
                  ('SHIPS (heads)', .11, .29), ('Neuron', .05, .00)]:
    chk(f'transfer {fam} transfer',
        av([r for r in prim if r['model'] == 'olmo3_7b_base' and FAM[r['condition']] == fam], 'P'), t)
    chk(f'transfer {fam} self',
        av([r for r in prim if r['model'] == 'olmo3_7b_base_own' and FAM[r['condition']] == fam], 'P'), s)

# gap stats
ARMS = ST + ['olmo3_7b_base', 'olmo3_7b_base_own']
pg = [r for r in prim if r['model'] in ARMS]
gaps = [(r, r['U_covert'] - r['O_overwarn']) for r in pg
        if isinstance(r.get('U_covert'), (int, float)) and isinstance(r.get('O_overwarn'), (int, float))]
checks.append((len(pg) == 100, 'primary grid = 100 cells', len(pg), 100))
checks.append((len(gaps) == 58, 'cells with defined gap = 58', len(gaps), 58))
checks.append((sum(1 for _, g in gaps if g > 0) == 0, 'positive-gap cells = 0',
               sum(1 for _, g in gaps if g > 0), 0))
for fam, v in [('Steering', -.124), ('Directional ablation', -.164), ('Neuron', -.100), ('SHIPS (heads)', -.114)]:
    chk(f'gap mean {fam}', mean([g for r, g in gaps if FAM[r['condition']] == fam]), v, tol=0.0015)
w = min(gaps, key=lambda t: t[1])
chk('most negative gap', w[1], -.389, tol=0.0015)
checks.append((w[0]['model'] == 'llama31_8b_control' and w[0]['condition'] == 'ships_top8',
               'most negative is Llama SHIPS', f"{w[0]['model']}/{w[0]['condition']}", 'llama/ships_top8'))

# base-arm facts
for m in ['olmo3_7b_base', 'olmo3_7b_base_own']:
    cr = av([r for r in prim if r['model'] == m], 'clean_rate')
    chk(f'{m} mean clean_rate ~0.20', cr, 0.21, tol=0.03)
sub = [r for r in prim if r['model'] == 'olmo3_7b_base' and r['dataset'] == 'jbb']
checks.append((sum(1 for r in sub if r.get('P') == 0.0) == 5, 'OLMo-Base jbb P==0 count = 5',
               sum(1 for r in sub if r.get('P') == 0.0), 5))
hacs = [r['hac'] for r in sub if isinstance(r.get('hac'), (int, float))]
chk('OLMo-Base jbb HAC min', min(hacs), .33, tol=0.006)
chk('OLMo-Base jbb HAC max', max(hacs), .76, tol=0.006)
ps = [r['P'] for r in sub if isinstance(r.get('P'), (int, float))]
chk('OLMo-Base jbb P max', max(ps), .47, tol=0.006)

# ---- Sec. cot-reasoning / cot-other: safety-reasoning judge (reasoning_metrics) ----
import os
_rm = 'runs/direction_a_v6/reports/reasoning_metrics.json'
if os.path.exists(_rm):
    _d = json.load(open(_rm))['rows']
    PRIM_SR = ['baseline'] + [c for c in FAM]
    for _m, _rate, _cov in [('qwen3_8b', 0.89, 0.30), ('olmo3_7b_think', 0.96, 0.32)]:
        _v = [r['safety_reasoning_rate'] for r in _d
              if r['model'] == _m and r['condition'] in PRIM_SR
              and isinstance(r.get('safety_reasoning_rate'), (int, float))]
        chk(f'SR verbalization rate {_m}', mean(_v) if _v else None, _rate)
        _c = []
        for r in _d:
            if r['model'] != _m or r['condition'] not in PRIM_SR:
                continue
            sc, npc, npr = (r.get('sr_mean_sentence_count'), r.get('n_pathway_completions'),
                            r.get('n_pathway_prefix_rows'))
            if isinstance(sc, (int, float)) and npc and npr:
                _c.append(sc / (npr / npc))
        chk(f'SR span coverage {_m}', mean(_c) if _c else None, _cov)
    _miss = [r for r in _d if r.get('safety_reasoning_rate') is None]
    checks.append((not _miss, 'all explicit cells have an SR rate', len(_miss), 0))

bad = [c for c in checks if not c[0]]
for ok, claim, got, wantv in checks:
    if not ok:
        g = f'{got:.4f}' if isinstance(got, float) else got
        print(f'  MISMATCH  {claim:44s} paper={wantv}  data={g}')
print(f'\n{len(checks)-len(bad)}/{len(checks)} paper numbers verified against the v6 bundle')
if bad:
    print(f'{len(bad)} MISMATCHES above need fixing')
