"""Export small review artifacts to the shared Windows workspace."""
from pathlib import Path
import json
import shutil

source=Path.home()/"vla_workspace/vla_interview_project/outputs_recent"
destination=Path("/mnt/c/VLATest/recent_results")
destination.mkdir(exist_ok=True)
for path in source.iterdir():
    if path.is_file() and path.suffix in {".md",".csv",".json",".png"}:
        shutil.copy2(path,destination/path.name)
video_links=[]
for episode in (5,6):
    for method in ("base10","rerank10"):
        folder=source/"heldout"/method/"libero_spatial_task0_seed3000"
        result=json.loads((folder/"eval_info.json").read_text())
        clip=Path(result["overall"]["video_paths"][episode])
        filename=f"explore_task0_episode{episode}_{method}.mp4"
        shutil.copy2(clip,destination/filename)
        video_links.append(f"- [{method}, task 0, episode {episode}]({filename})")
chosen=json.loads((source/"selected_confirmation.json").read_text())["method"]
for task in (0,1):
    pair=[]
    for method in ("base10",chosen):
        folder=source/"confirmation"/method/f"libero_spatial_task{task}_seed4000"
        pair.append(json.loads((folder/"eval_info.json").read_text()))
    a,b=[d["per_task"][0]["metrics"]["successes"] for d in pair]
    different=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    for episode in different:
        for method,data in zip(("base10",chosen),pair):
            filename=f"confirm_task{task}_episode{episode}_{method}.mp4"
            shutil.copy2(data["overall"]["video_paths"][episode],destination/filename)
            success=data["per_task"][0]["metrics"]["successes"][episode]
            video_links.append(f"- [confirmation {method}, task {task}, episode {episode}, success={success}]({filename})")
report=destination/"FINAL_REPORT.md"
report.write_text(report.read_text()+"\n\n## 导出的证据视频\n\n"+"\n".join(video_links)+"\n",encoding="utf-8")
print(destination)
print("report",report)
