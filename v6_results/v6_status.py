from pathlib import Path
import json
r=Path('/home/xiaolongwu/vla_workspace/outputs/expert_continuation_v6')
for p in sorted(r.glob('E*_*_bs24_w2_*')):
    f=p/'steps.jsonl'
    rows=[json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
    result=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else {}
    print(p.name, result.get('status','running'),len(rows),
          'loss',round(rows[-1]['loss'],4) if rows else None,
          'allocated',round(max(x['peak_allocated_gib'] for x in rows),3) if rows else None,
          'reserved',round(max(x['peak_reserved_gib'] for x in rows),3) if rows else None, flush=True)
print((r/'pipeline.log').read_text()[-1300:],flush=True)
if (r/'extension.log').exists():print((r/'extension.log').read_text()[-1300:],flush=True)
