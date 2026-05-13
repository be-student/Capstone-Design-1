import json

with open('_test_results/iter12/coverage.json') as f:
    cov = json.load(f)
files = cov['files']

targets = [
    'src\\dashboard\\app.py',
    'src\\main.py',
    'src\\dashboard\\calculations.py',
    'src\\dashboard\\data_loader.py',
    'src\\streaming\\redis_producer.py',
    'src\\models\\churn_model.py',
]

for t in targets:
    info = files.get(t)
    if not info:
        print('MISSING file key:', t)
        continue
    miss = info.get('missing_lines', [])
    ranges = []
    cur_s = cur_e = None
    for ln in miss:
        if cur_s is None:
            cur_s = cur_e = ln
        elif ln == cur_e + 1:
            cur_e = ln
        else:
            ranges.append((cur_s, cur_e))
            cur_s = cur_e = ln
    if cur_s is not None:
        ranges.append((cur_s, cur_e))
    ranges_by_size = sorted(ranges, key=lambda r: -(r[1] - r[0]))[:8]
    print()
    print(t.replace('\\', '/'))
    for s, e in ranges_by_size:
        print(f'  miss {s}-{e}  (size {e - s + 1})')
