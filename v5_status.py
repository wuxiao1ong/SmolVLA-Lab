from pathlib import Path
import json
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5')
for run in sorted(root.glob('*_20*')):
 p=run/'steps.jsonl'
 if not p.exists():continue
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 r=rows[-1];result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {}
 print(json.dumps(dict(run=run.name,status=result.get('status','running'),step=r['step'],last100_loss=sum(x['loss'] for x in rows[-100:])/len(rows[-100:]),lr=r['lr'],peak_allocated=max(x['peak_allocated_gib'] for x in rows),checkpoints=[p.name for p in (run/'train/checkpoints').glob('*') if p.name.isdigit()]),ensure_ascii=False))
