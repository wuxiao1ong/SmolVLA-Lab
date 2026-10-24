from pathlib import Path
import json
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');p=root/'metadata/selected_batch.json';d=json.loads(p.read_text());d['selection']='batch32 reserved7.52GiB too close; batch24 passed1/3steps, 3-step peak allocated6.093/reserved6.266GiB;workers2';d['peak_allocated_gib_check3']=6.092993259429932;d['peak_reserved_gib_check3']=6.265625;d['steady_samples_per_s_check3']=30.420996223779348;p.write_text(json.dumps(d,indent=2))
for p in sorted(root.glob('A_formal2000_bs24*')):
 if (p/'result.json').exists():print((p/'result.json').read_text());continue
 if (p/'steps.jsonl').exists():
  lines=(p/'steps.jsonl').read_text().splitlines();r=json.loads(lines[-1]);print(json.dumps(dict(run=p.name,step=r['step'],loss=r['loss'],lr=r['lr'],memory_allocated=r['peak_allocated_gib'],memory_reserved=r['peak_reserved_gib'],samples_per_s=24/(r['update_s']+r['data_s']))))
