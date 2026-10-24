from pathlib import Path
import json,subprocess,hashlib
import torch,peft,transformers,platform
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4');a=root/'audit'
summary=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,peft=peft.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,compute_capability=torch.cuda.get_device_capability(),lerobot_commit=subprocess.check_output(['git','-C','/home/xiaolongwu/vla_workspace/lerobot','rev-parse','HEAD'],text=True).strip(),lerobot_status=subprocess.check_output(['git','-C','/home/xiaolongwu/vla_workspace/lerobot','status','--short'],text=True),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')})
(a/'environment.json').write_text(json.dumps(summary,indent=2))
(a/'nvidia_smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
print('ENV_AUDIT_PASS',summary['peft'],repr(summary['lerobot_status']))
