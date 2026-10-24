from pathlib import Path
import json,statistics
r=Path('/home/xiaolongwu/vla_workspace/outputs/two_vlm_layers_v8')
for p in sorted(r.glob('*_*_bs*_w2_*')):
    rows=[json.loads(l) for l in (p/'steps.jsonl').read_text().splitlines()] if (p/'steps.jsonl').exists() else []
    result=json.loads((p/'result.json').read_text()) if (p/'result.json').exists() else {}
    print(p.name,result.get('status','running'),'steps',len(rows),'loss_last100',round(statistics.mean(x['loss'] for x in rows[-100:]),4) if rows else None,'allocated',round(max(x['peak_allocated_gib'] for x in rows),3) if rows else None,'reserved',round(max(x['peak_reserved_gib'] for x in rows),3) if rows else None,'vlm', {k:v for k,v in rows[-1]['categories'].items() if k.startswith('vlm')} if rows else None,flush=True)
print((r/'pipeline.log').read_text()[-900:],flush=True)


