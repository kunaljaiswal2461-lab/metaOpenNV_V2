import json
with open('out.txt') as f:
    lines = [json.loads(l) for l in f if l.strip()]
ends = [l for l in lines if l['event'] == '[END]']
for e in ends:
    print(e)
