from pathlib import Path
import shutil

source = Path(__file__).resolve().parent
destination = Path.home() / "vla_workspace/vla_interview_project/recent_methods"
destination.mkdir(exist_ok=True)
for path in source.iterdir():
    if path.is_file() and path.suffix in {".py", ".md", ".json"} and path.resolve() != (destination / path.name).resolve():
        shutil.copy2(path, destination / path.name)
print(destination)
