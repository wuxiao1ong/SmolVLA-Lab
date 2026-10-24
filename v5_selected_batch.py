from pathlib import Path
import json
root=Path('/home/xiaolongwu/vla_workspace/outputs/base_single_task_v5');run=sorted(root.glob('A_check3_bs24_w2_*'))[-1];rows=[json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()]
print(json.dumps(dict(result=json.loads((run/'result.json').read_text()),steady_samples_per_s=24/(sum(x['update_s']+x['data_s'] for x in rows[1:])/2),steady_update_s=sum(x['update_s'] for x in rows[1:])/2,steady_data_s=sum(x['data_s'] for x in rows[1:])/2),indent=2))
(root/'metadata/selected_batch.json').write_text(json.dumps(dict(batch=24,num_workers=2,steps=2000,effective_sample_budget=48000,equivalent_coverage=48000/5472,lr=1e-4,warmup=100,selection='32 passed but reserved7.52GiB too close;24 passed1/3steps with reserved5.86GiB; workers2 steady throughput recorded',gate_run=str(run)),indent=2))
