"""Build a focused report from the completed v3 LIBERO runs in WSL."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path


VLA_ROOT = Path.home() / "vla_workspace"
ROOT = VLA_ROOT / "vla_interview_project" / "outputs"
MODEL_SHA = "a24f1f380461b665158d52acd140152c79e8b36d"
CONDITIONS = (("fixed05", 5), ("fixed10", 10), ("fixed25", 25), ("fixed50", 50), ("adapt_q50", None))


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    proportion = wins / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return 100 * (center - spread), 100 * (center + spread)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    pilot = json.loads((VLA_ROOT / "audit" / "v3_pilot_q50_threshold.json").read_text())
    threshold = float(pilot["threshold"])
    episode_rows: list[dict] = []
    efficiency_rows: list[dict] = []
    horizons: dict[str, list[int]] = defaultdict(list)

    for condition, fixed_horizon in CONDITIONS:
        for task_id in (0, 1):
            folder = ROOT / condition / f"libero_object_task{task_id}_seed2000"
            manifest = json.loads((folder / "manifest.json").read_text())
            evaluation = json.loads((folder / "eval_info.json").read_text())
            successes = evaluation["per_task"][0]["metrics"]["successes"]
            events = [json.loads(line) for line in (folder / "planner.jsonl").read_text().splitlines() if line]
            chunks = [event for event in events if event.get("event") == "chunk"]
            videos = evaluation["overall"]["video_paths"]

            assert manifest["policy"].endswith(MODEL_SHA)
            assert manifest["seed"] == 2000 and manifest["task_id"] == task_id
            assert manifest["episodes"] == len(successes) == len(videos) == 10
            assert all(Path(video).is_file() for video in videos)
            assert chunks
            allowed = {5, 25} if fixed_horizon is None else {fixed_horizon}
            assert {chunk["horizon"] for chunk in chunks} <= allowed
            if fixed_horizon is None:
                assert abs(manifest["threshold"] - threshold) < 1e-12

            for episode_idx, success in enumerate(successes):
                episode_rows.append(
                    dict(
                        condition=condition,
                        suite="libero_object",
                        task_id=task_id,
                        episode_idx=episode_idx,
                        seed=2000 + episode_idx,
                        success=int(success),
                    )
                )
            horizons[condition].extend(chunk["horizon"] for chunk in chunks)
            efficiency_rows.append(
                dict(
                    condition=condition,
                    task_id=task_id,
                    successes=sum(successes),
                    episodes=len(successes),
                    calls=len(chunks),
                    calls_per_episode=len(chunks) / len(successes),
                    mean_horizon=sum(chunk["horizon"] for chunk in chunks) / len(chunks),
                    eval_s=evaluation["overall"]["eval_s"],
                )
            )

    assert len(episode_rows) == 100 and len(efficiency_rows) == 10
    write_csv(ROOT / "formal_episodes.csv", episode_rows)
    write_csv(ROOT / "formal_efficiency.csv", efficiency_rows)

    native = json.loads((VLA_ROOT / "outputs" / "v3_native_fixed10_20261020_01" / "eval_info.json").read_text())
    assert native["overall"]["n_episodes"] == 1
    smoke = {}
    for name in ("smoke_fixed10", "smoke_fixed25", "smoke_adaptive"):
        path = ROOT / name / "libero_object_task0_seed1000" / "eval_info.json"
        smoke[name] = sum(json.loads(path.read_text())["per_task"][0]["metrics"]["successes"])

    lines = [
        "# v3 第 1–3 章实测结果（2026-10-20）",
        "",
        "环境：LeRobot v0.6.1（7e241bd630a3719a56157a497ce5d08f244784f1），PyTorch 2.10.0+cu128，RTX 5060 Laptop 8GB。",
        f"官方模型 revision：`{MODEL_SHA}`。",
        "",
        "正式配置：LIBERO Object task 0/1、相对动作、20 Hz、固定初始状态、batch 1、seed 2000，每个 task 每种条件 10 个 episode。固定与自适应方法均先采样 50 步动作，再由同一个控制器截断实际执行长度。",
        "",
        "## 前置检查",
        "",
        f"- 原生 fixed10：task 0，seed 2000，{int(native['overall']['avg_sum_reward'])}/1 成功；JSON 与 MP4 已保存。",
        "- 控制器单元测试：3/3 通过；run_one.sh 语法检查通过。",
        "- 三个 Smoke Test：fixed10、fixed25、自适应均正常结束，JSON、manifest、planner trace 齐全；成功数分别为 "
        + ", ".join(f"{name} {wins}/1" for name, wins in smoke.items()) + "。",
        f"- 独立 Pilot：seed 1100，5 个 episode，{pilot['n_chunks']} 次采样；分数中位数阈值 `{threshold:.9f}`，未用成功率选阈值。",
        "",
        "## 正式探索性消融",
        "",
        "| 策略 | task 0 | task 1 | 合计成功率 | Wilson 95% 区间 | 模型采样/episode | 平均计划执行步数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, _ in CONDITIONS:
        rows = [row for row in efficiency_rows if row["condition"] == condition]
        wins = sum(row["successes"] for row in rows)
        total = sum(row["episodes"] for row in rows)
        calls = sum(row["calls"] for row in rows)
        lower, upper = wilson(wins, total)
        lines.append(
            f"| {condition} | {rows[0]['successes']}/10 | {rows[1]['successes']}/10 | "
            f"{wins}/{total} ({100 * wins / total:.0f}%) | {lower:.1f}–{upper:.1f}% | "
            f"{calls / total:.1f} | {sum(horizons[condition]) / len(horizons[condition]):.1f} |"
        )
    lines += [
        "",
        "自适应组的 5/25 步分支来自真实 trace：task 0 为 55/50 次，task 1 为 89/52 次。",
        "",
        "配对比较：自适应相对 fixed10 为 −10 个百分点（20 个配对 episode 的任务分层 bootstrap 95% 区间 −25 到 0）；相对 fixed5 和 fixed25 均为 +5 个百分点（各自区间 −15 到 +25）。这些区间和检验仅供探索，样本量不足以支持稳定收益结论。",
        "",
        "当前数据中 fixed10 为 20/20，自适应为 18/20；自适应总采样 246 次，fixed10 为 270 次，约少 9%，但 task 1 上自适应反而调用更多。第 3 章的 240 episode hold-out 以“初步有价值”为前提；本轮没有显示清晰的成功率或开销优势，因此未启动。",
        "",
        "EGL 评测可运行，日志仍出现 kms_swrast 软件渲染回退；模型策略推理使用 CUDA。",
        "",
        "来源：本目录 formal_episodes.csv、formal_efficiency.csv，各条件的 eval_info.json、manifest.json、planner.jsonl 与 MP4。",
    ]
    report = ROOT / "v3_ch1to3_results.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    print("formal episodes:", len(episode_rows), "runs:", len(efficiency_rows))


if __name__ == "__main__":
    main()
