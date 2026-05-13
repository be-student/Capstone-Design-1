import json
from collections import defaultdict

with open('_test_results/iter12/coverage.json') as f:
    cov = json.load(f)

files = cov['files']
totals = cov['totals']

print('=== TOTALS ===')
print('covered_lines:', totals['covered_lines'])
print('num_statements:', totals['num_statements'])
print('missing_lines:', totals['missing_lines'])
print(f'percent_covered: {totals["percent_covered"]:.2f}')
print('file_count:', len(files))
print()


def norm(p):
    return p.replace('\\', '/')


def domain_of(p):
    p = norm(p)
    if '/dashboard/' in p or p.endswith('dashboard'):
        return 'dashboard'
    if '/monitoring/' in p:
        return 'monitoring'
    if '/models/' in p:
        return 'models'
    if '/analysis/' in p:
        return 'analysis'
    if '/optimization/' in p:
        return 'optimization'
    if '/features/' in p:
        return 'features'
    if '/streaming/' in p:
        return 'streaming'
    if '/pipeline/' in p:
        return 'pipeline'
    if '/data/' in p:
        return 'data'
    return 'core'


agg = defaultdict(lambda: {'files': 0, 'stmt': 0, 'cov': 0, 'miss': 0})
for path, info in files.items():
    d = domain_of(path)
    agg[d]['files'] += 1
    agg[d]['stmt'] += info['summary']['num_statements']
    agg[d]['cov'] += info['summary']['covered_lines']
    agg[d]['miss'] += info['summary']['missing_lines']

print('| Domain | Files | LOC | Covered | % |')
print('|---|---:|---:|---:|---:|')
for d, v in sorted(agg.items()):
    pct = (v['cov'] / v['stmt'] * 100) if v['stmt'] else 0
    print(f'| {d} | {v["files"]} | {v["stmt"]} | {v["cov"]} | {pct:.1f}% |')

print()
print('=== HIGH (>=90%) ===')
high = []
for path, info in files.items():
    s = info['summary']
    if s['num_statements'] == 0:
        continue
    pct = s['percent_covered']
    if pct >= 90:
        high.append((path, pct, s['num_statements'], s['missing_lines']))
for r in sorted(high, key=lambda x: -x[1]):
    print(f'  {r[1]:5.1f}% | {r[2]:5d} stmt | {r[3]:5d} miss | {norm(r[0])}')

print()
print('=== MID (70-89.99%) ===')
mid = []
for path, info in files.items():
    s = info['summary']
    if s['num_statements'] == 0:
        continue
    pct = s['percent_covered']
    if 70 <= pct < 90:
        mid.append((path, pct, s['num_statements'], s['missing_lines']))
for r in sorted(mid, key=lambda x: x[1]):
    print(f'  {r[1]:5.1f}% | {r[2]:5d} stmt | {r[3]:5d} miss | {norm(r[0])}')

print()
print('=== LOW (<70%) ===')
low = []
for path, info in files.items():
    s = info['summary']
    if s['num_statements'] == 0:
        continue
    pct = s['percent_covered']
    if pct < 70:
        low.append((path, pct, s['num_statements'], s['missing_lines'], info.get('missing_lines', [])))
for r in sorted(low, key=lambda x: x[1]):
    print(f'  {r[1]:5.1f}% | {r[2]:5d} stmt | {r[3]:5d} miss | {norm(r[0])}')

print()
print('=== FILES OF INTEREST ===')
interest = [
    'src/dashboard/app.py',
    'src/dashboard/data_loader.py',
    'src/dashboard/calculations.py',
    'src/dashboard/monitoring_view.py',
    'src/dashboard/recommendations_view.py',
    'src/dashboard/system_health_view.py',
    'src/dashboard/utils/dashboard_helpers.py',
    'src/models/churn_model.py',
    'src/models/dl_trainer.py',
    'src/models/clv_model.py',
    'src/models/uplift_model.py',
    'src/models/recommendations.py',
    'src/models/scoring_api.py',
    'src/models/survival_analysis.py',
    'src/models/shap_explainer.py',
    'src/models/mlflow_tracking.py',
    'src/main.py',
    'src/analysis/cohort_analysis.py',
    'src/analysis/ab_testing.py',
    'src/optimization/budget_optimizer.py',
    'src/monitoring/drift_detection.py',
    'src/monitoring/ks_drift.py',
    'src/monitoring/monitoring_service.py',
]
norm_files = {norm(p): info for p, info in files.items()}
for k in interest:
    info = norm_files.get(k)
    if info is None:
        print(f'  ??? | {k}')
        continue
    s = info['summary']
    print(f'  {s["percent_covered"]:5.1f}% | {s["num_statements"]:5d} stmt | {s["missing_lines"]:5d} miss | {k}')
