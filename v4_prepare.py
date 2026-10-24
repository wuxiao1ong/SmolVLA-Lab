from pathlib import Path
import zipfile, json, hashlib, subprocess
root=Path('/home/xiaolongwu/vla_workspace/vla_interview_project/vla_training_ablation_v4')
root.mkdir(exist_ok=False)
z=zipfile.ZipFile('/mnt/d/Downloads/vla_training_ablation_v4.zip')
for x in z.infolist():
 p=root/x.filename
 assert p.resolve().is_relative_to(root.resolve())
 if not x.is_dir(): p.write_bytes(z.read(x))
(root/'archive_sha256.txt').write_text(hashlib.sha256(Path('/mnt/d/Downloads/vla_training_ablation_v4.zip').read_bytes()).hexdigest())
a=root/'audit'; a.mkdir()
(a/'pip_freeze.txt').write_text(subprocess.check_output(['/home/xiaolongwu/vla_workspace/env/bin/python','-m','pip','freeze'],text=True))
print(root)
