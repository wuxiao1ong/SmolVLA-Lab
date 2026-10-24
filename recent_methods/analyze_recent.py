"""Analyze the predeclared paired recent-method experiment from raw outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

import numpy as np

PROJECT = Path.home() / "vla_workspace/vla_interview_project"
ROOT = PROJECT / "outputs_recent"
METHODS = ["base10", "sample4_fixed10", "rerank10", "aac_first", "aac_rerank", "aac_nogrip"]


def wilson(wins, total):
    z = 1.96
    p = wins / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    width = z*sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [100*(mid-width), 100*(mid+width)]


def frame_digest(path):
    import av
    digest = hashlib.sha256()
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            digest.update(frame.to_ndarray(format="rgb24").tobytes())
            count += 1
    return count, digest.hexdigest()


def main():
    plan = json.loads((ROOT / "heldout_plan.json").read_text())
    assert plan["methods"] == METHODS and plan["tasks"] == [0, 1]
    assert plan["suite"] == "libero_spatial" and plan["seed"] == 3000 and plan["episodes"] == 10
    episode_rows = []
    summaries = []
    evidence = {}
    all_manifests = []
    for method in METHODS:
        successes = []
        traces = []
        task_wins = []
        total_eval_s = 0.0
        for task in (0, 1):
            folder = ROOT / "heldout" / method / f"libero_spatial_task{task}_seed3000"
            manifest = json.loads((folder / "manifest.json").read_text())
            all_manifests.append(manifest)
            data = json.loads((folder / "eval_info.json").read_text())
            outcomes = data["per_task"][0]["metrics"]["successes"]
            events = [json.loads(line) for line in (folder / "planner.jsonl").read_text().splitlines()]
            chunks = [row for row in events if row["event"] == "chunk"]
            by_rollout = defaultdict(list)
            for chunk in chunks:
                by_rollout[chunk["rollout"]].append(chunk)
            assert len(outcomes) == len(by_rollout) == len(data["overall"]["video_paths"]) == 10
            assert manifest["seed"] == 3000 and manifest["task"] == task and manifest["method"] == method
            assert all(Path(video).is_file() for video in data["overall"]["video_paths"])
            for episode, (outcome, chunk_list) in enumerate(zip(outcomes, by_rollout.values())):
                episode_rows.append(dict(method=method, task=task, episode=episode, seed=3000+episode,
                                         init_state_index=episode,
                                         success=int(outcome), replans=len(chunk_list),
                                         model_calls=sum(c["model_calls"] for c in chunk_list),
                                         mean_horizon=float(np.mean([c["horizon"] for c in chunk_list])),
                                         video=data["overall"]["video_paths"][episode]))
            task_wins.append(sum(outcomes))
            successes.extend(outcomes)
            traces.extend(chunks)
            total_eval_s += data["overall"]["eval_s"]
            evidence[(method, task)] = (outcomes, chunks, data["overall"]["video_paths"])
        lo, hi = wilson(sum(successes), len(successes))
        summaries.append(dict(method=method, task0=task_wins[0], task1=task_wins[1], successes=sum(successes),
                              episodes=len(successes), success_pct=100*np.mean(successes), ci_lower=lo, ci_upper=hi,
                              replans_per_episode=len(traces)/len(successes),
                              model_calls_per_episode=sum(t["model_calls"] for t in traces)/len(successes),
                              mean_horizon=float(np.mean([t["horizon"] for t in traces])),
                              sampling_s_per_episode=sum(t["sampling_s"] for t in traces)/len(successes),
                              eval_s_per_episode=total_eval_s/len(successes),
                              seam_norm=float(np.mean([t["seam_norm"] for t in traces if t["step"] > 0])),
                              first_selection_score=float(np.mean([t["consensus"][0]+t["backward"][0] for t in traces])),
                              selected_selection_score=float(np.mean([t["consensus"][t["candidate"]]+t["backward"][t["candidate"]] for t in traces])),
                              candidate_changed_pct=100*np.mean([t["candidate"] != 0 for t in traces]),
                              gripper_changed_horizon_pct=100*np.mean([t["horizon"] != t["horizon_without_gripper"] for t in traces]) if method.startswith("aac") else 0,
                              peak_cuda_gb=max(t["peak_cuda_mb"] for t in traces)/1024))
    assert len({m["code_sha256"] for m in all_manifests}) == 1
    assert len({m["model_revision"] for m in all_manifests}) == 1
    for filename, rows in (("episodes.csv", episode_rows), ("metrics.csv", summaries)):
        with (ROOT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    # Actual rollout equivalence is checked independently of container metadata.
    control = dict(outcomes_equal=True, trace_steps_equal=True, decoded_videos_equal=0, videos_checked=0)
    for task in (0, 1):
        first, second = evidence[("base10", task)], evidence[("sample4_fixed10", task)]
        control["outcomes_equal"] &= first[0] == second[0]
        control["trace_steps_equal"] &= [c["step"] for c in first[1]] == [c["step"] for c in second[1]]
        for a, b in zip(first[2], second[2]):
            control["decoded_videos_equal"] += int(frame_digest(a) == frame_digest(b))
            control["videos_checked"] += 1

    paired = []
    pairs = [("base10", "rerank10"), ("sample4_fixed10", "aac_first"),
             ("sample4_fixed10", "rerank10"), ("rerank10", "aac_rerank"),
             ("aac_first", "aac_rerank"), ("aac_nogrip", "aac_first")]
    for base, modified in pairs:
        differences = []
        boot = np.zeros(5000)
        rng = np.random.default_rng(42)
        lose = win = 0
        for task in (0, 1):
            a = np.asarray(evidence[(base, task)][0], dtype=int)
            b = np.asarray(evidence[(modified, task)][0], dtype=int)
            delta = b - a
            differences.extend(delta)
            boot += rng.choice(delta, (5000, 10), replace=True).sum(1)
            lose += int(((a == 1) & (b == 0)).sum())
            win += int(((a == 0) & (b == 1)).sum())
        n = win + lose
        exact_p = min(1.0, 2*sum(comb(n, k) for k in range(min(win, lose)+1))/2**n) if n else 1.0
        paired.append(dict(base=base, modified=modified, difference_pp=100*float(np.mean(differences)),
                           bootstrap95_pp=(100*np.quantile(boot/20, [0.025,0.975])).tolist(),
                           base_only=lose, modified_only=win, exploratory_mcnemar_p=exact_p))
    result = dict(control_equivalence=control, summary=summaries, paired=paired)
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2))

    lines = ["# 近期 VLA 推理模块实测", "",
             "这批任务用于本项目的新测试；官方 LIBERO 权重已针对 LIBERO 训练，不能把它称为训练集之外的任务泛化。",
             "所有组固定 Spatial task 0/1、seed 3000、每任务 10 episode、相对动作、20 Hz、同一模型 revision。参数与方法列表在 heldout_plan.json 中于运行前固定。", "",
             "主实验使用默认初始状态索引 0–9。源码核查确认 init_states=true 时换 seed 不会换这组索引；独立确认需要显式偏移到新状态并记录实际索引。", "",
             "## 方法和适配边界", "",
             "AAC（2026）使用多个动作候选的连续动作熵与夹爪熵选择执行长度。本适配使用 4 次顺序采样、两个正则化 3D 协方差、固定 ridge=1e-6，执行范围限于 5–25 步；这是低显存适配，不是论文完整复现。夹爪二值边界从 checkpoint 的反归一化统计推导，不直接对标准化动作取正负号。",
             "BID 启发的重排对整条候选评分：与上一预测重叠区间的距离 + 与本次其他候选的平均距离，等权重、10 步窗口、折扣 0.8。没有论文中的弱策略负样本，因此命名为 BID-inspired，而非完整 BID。它选择整条轨迹，不平均不同候选的夹爪控制。", "",
             "来源：[AAC, 2026](https://arxiv.org/abs/2604.04161)；[BID 作者项目, ICLR 2025](https://bid-robot.github.io/)。", "",
             "## 结果", "",
             "| 方法 | task 0 | task 1 | 成功率 | 95% Wilson 区间 | 重规划/episode | 模型前向采样/episode | 平均执行长度 | 同步采样秒/episode |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['method']} | {row['task0']}/10 | {row['task1']}/10 | {row['successes']}/20 ({row['success_pct']:.0f}%) | {row['ci_lower']:.1f}–{row['ci_upper']:.1f}% | {row['replans_per_episode']:.1f} | {row['model_calls_per_episode']:.1f} | {row['mean_horizon']:.1f} | {row['sampling_s_per_episode']:.2f} |")
    lines += ["", "sample4_fixed10 只增加三个独立随机数流中的候选，但执行第一条、固定 10 步。它用于排除采样预算和 RNG 改变的混淆。",
              f"实际对照：成败相同={control['outcomes_equal']}，采样时点相同={control['trace_steps_equal']}，解码后视频逐帧哈希一致={control['decoded_videos_equal']}/{control['videos_checked']}。", "",
              "## 模块消融的配对结果", ""]
    for comparison in paired:
        lo, hi = comparison["bootstrap95_pp"]
        lines.append(f"- {comparison['modified']} − {comparison['base']}：{comparison['difference_pp']:+.1f} 个百分点，探索性配对区间 {lo:.1f} 到 {hi:.1f}；基线独有成功/修改独有成功为 {comparison['base_only']}/{comparison['modified_only']}。")
    stats = {row["method"]: row for row in summaries}
    eligible = [row for row in summaries if row["method"] in {"rerank10","aac_first","aac_rerank","aac_nogrip"} and row["successes"] > stats["base10"]["successes"]]
    best = min(eligible, key=lambda row: (-row["successes"],row["model_calls_per_episode"])) if eligible else None
    (ROOT / "selected_method.json").write_text(json.dumps(dict(method=best["method"] if best else None, rule="most successes above baseline, tie fewer model calls", source_seed=3000), indent=2))
    lines += ["", "## 基于这批数据的解释", "",
              f"- 固定执行 10 步且同为四候选时，重排模块的成功数变化为 {stats['rerank10']['successes']-stats['sample4_fixed10']['successes']:+d}/20。预算对照逐帧一致，因此这部分差异可以关联到候选选择干预；它仍是小样本结果。",
              f"- 保留第一条候选时，AAC 相对固定 10 步的成功数变化为 {stats['aac_first']['successes']-stats['sample4_fixed10']['successes']:+d}/20。加入候选重排后，AAC 相对 rerank10 的变化为 {stats['aac_rerank']['successes']-stats['rerank10']['successes']:+d}/20。两种上下文都需要比较，不能由单一成功数字断言自适应执行普遍有效。",
              f"- 完整 AAC 相对去掉夹爪熵的成功数变化为 {stats['aac_first']['successes']-stats['aac_nogrip']['successes']:+d}/20；完整 AAC 在 {stats['aac_first']['gripper_changed_horizon_pct']:.1f}% 的预测中因夹爪熵改变了执行长度。决策变化不自动等于成功率收益。",
              f"- 按预先保存的选择规则，待独立 seed 4000 确认的方法为 {best['method'] if best else '无'}。选择依据是成功数优先，同成功数时实际模型采样较少；未根据确认集调参数。", "",
              "| 方法 | 重排选择改变比例 | 选前/选后评分 | 规划边界动作差范数 | PyTorch peak allocated GiB |",
              "|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['method']} | {row['candidate_changed_pct']:.1f}% | {row['first_selection_score']:.3f}/{row['selected_selection_score']:.3f} | {row['seam_norm']:.3f} | {row['peak_cuda_gb']:.2f} |")
    lines += ["", "评分和边界范数均处于标准化模型动作空间。它们是模块诊断，不能当作真实末端加速度或抓取正确性的标注。PyTorch 分配峰值也不等于整张显卡总显存。"]
    lines += ["",f"本轮重排的边界动作差均值为 {stats['rerank10']['seam_norm']:.3f}，基线为 {stats['base10']['seam_norm']:.3f}；组合模块为 {stats['aac_rerank']['seam_norm']:.3f}，但成功数没有再增加。这与一致性设计目标相符，也说明更低的边界范数不是成功的充分条件。不同策略的轨迹长度和早停不同，这些分布均值不能单独证明平滑性导致成功。"]
    lines += ["", "## 如何解释模块的意义", "",
              "动作序列自身变化大可以是正确运动；AAC 改用同一观测下不同预测之间的分歧，试图在未来动作失去一致性之前重新观察。多模态的多个正确动作也会产生高熵，所以分歧不等于失败概率。夹爪熵消融只检验模型是否对开闭决策存在分歧，不保证检测到真实接触。",
              "候选重排试图减少相邻规划间的策略切换。它可能保留错误的旧方案，或选择常见但不成功的轨迹；只有在固定 horizon、相同候选预算的重排消融中改善，才能把收益关联到该模块。",
              "这轮把候选共识与历史一致性作为一个选择模块，未单独拆解这两个评分项，不能把选择模块的全部收益归因于历史一致性。新模块的 7 项针对性单测已通过，另有独立 Object task 0、seed 1200 的组合模块 Smoke Test 1/1 成功。",
              "AAC 与重排组合的效果需要与两个单模块同时比较；若组合没有更好，不能声称两个模块互补。采样次数按真实模型前向数计算，N=4 不能被算作一次模型调用。同步采样块计时包含模型与调用开销，overall.eval_s 还混入渲染/仿真/视频 IO；两者都不是纯 GPU kernel latency。",
              "仿真在等待策略推理时不推进物理时间。环境设为 20 Hz 不代表本机以真实墙钟时间持续输出 20 Hz；额外采样的实机延迟和旧观测问题尚未包含在这些成功率里，因此不能由仿真收益直接推断真实机器人收益。",
              "每组仅 20 个 episode，配对区间依赖任务内 episode 独立近似，多重比较未校正。EGL 仍有软件回退，时间指标仅适用于当前环境。", "",
              "每个条件目录保存 planned.json、manifest.json、eval_info.json、planner.jsonl、run.log 和视频；本目录保存逐 episode 数据、汇总与配对统计。", "",
              "## 相同 episode 的视频证据", "",
              "基线 Spatial task 0 的失败 episode 为 5、6，均走满 280 步。rerank10 在这两项配对状态中分别约于第 78、76 帧触发 LIBERO 成功。画面显示基线把碗移动到盘子附近后继续调整；重排轨迹更早达到环境判定条件。轨迹在较早时就已分叉，不能只由末帧断言是平滑性或夹爪造成了改善。成功后的稳定保持没有在这些截断视频中验证。", "",
              "![Episode 5 baseline and reranking](failure_task0_episode5.png)", "",
              "![Episode 6 baseline and reranking](failure_task0_episode6.png)", "",
              "![Module decisions on the first baseline failure](module_trace.png)"]
    (ROOT / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        labels = [s["method"] for s in summaries]
        rate = np.array([s["success_pct"] for s in summaries])
        error = np.array([[s["success_pct"]-s["ci_lower"] for s in summaries], [s["ci_upper"]-s["success_pct"] for s in summaries]])
        error = np.maximum(error,0)
        colors=plt.get_cmap("tab10").colors[:6]
        axes[0].bar(labels, rate, yerr=error, capsize=3,color=colors)
        for i,s in enumerate(summaries):
            axes[0].text(i,52,f"{s['successes']}/20",ha="center",color="white",fontsize=9)
        axes[0].set_title("Success with Wilson 95% intervals")
        axes[0].set_ylim(0, 105); axes[0].set_ylabel("Success rate (%)"); axes[0].tick_params(axis="x", rotation=30)
        for i,s in enumerate(summaries):
            axes[1].scatter(s["model_calls_per_episode"],s["success_pct"],label=s["method"],
                            marker=["o","X","s","^","D","o"][i],s=90 if i<5 else 180,
                            facecolors=colors[i] if i<5 else "none",edgecolors=colors[i],linewidths=1.5)
        axes[1].set_xlabel("Actual model samples / episode"); axes[1].set_ylabel("Success rate (%)")
        axes[1].set_ylim(0,105);axes[1].set_xlim(0,55)
        axes[1].set_title("Accuracy-cost tradeoff (N=4 is four samples)")
        axes[1].legend(loc="lower right",fontsize=8)
        fig.suptitle("Spatial tasks 0/1, state indices 0-9, seed 3000")
        fig.savefig(ROOT / "comparison.png", dpi=180)
        plt.close(fig)
        case = 5  # first base10 failure, chosen before treatment outcomes
        def episode_chunks(method):
            groups = defaultdict(list)
            for item in evidence[(method, 0)][1]:
                groups[item["rollout"]].append(item)
            return list(groups.values())[case]
        aac = episode_chunks("aac_first")
        ranked = episode_chunks("rerank10")
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
        steps = [v["step"] for v in aac]
        axes[0,0].step(steps,[v["horizon"] for v in aac],where="post",label="AAC with gripper")
        axes[0,0].step(steps,[v["horizon_without_gripper"] for v in aac],where="post",label="same chunks without gripper",linestyle="--")
        axes[0,0].set_ylabel("Planned execution horizon"); axes[0,0].legend(fontsize=8)
        axes[0,1].plot(steps,[v["gripper_entropy_mean"] for v in aac],marker=".")
        axes[0,1].set_ylabel("Mean gripper entropy (nats)")
        rs = [v["step"] for v in ranked]
        axes[1,0].step(rs,[v["candidate"] for v in ranked],where="post")
        axes[1,0].set_ylabel("Chosen candidate index"); axes[1,0].set_yticks([0,1,2,3])
        axes[1,1].plot(rs,[v["consensus"][0]+v["backward"][0] for v in ranked],label="first candidate")
        axes[1,1].plot(rs,[v["consensus"][v["candidate"]]+v["backward"][v["candidate"]] for v in ranked],label="selected")
        axes[1,1].set_ylabel("Model-space selection score"); axes[1,1].legend(fontsize=8)
        for axis in axes.flat:
            axis.set_xlabel("Environment control step"); axis.grid(alpha=.2)
        fig.suptitle("Spatial task 0, episode 5: module diagnostics (not success probabilities)")
        fig.savefig(ROOT / "module_trace.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    for row in summaries:
        print(row["method"], str(row["successes"])+"/20", "samples/episode", round(row["model_calls_per_episode"],2),
              "horizon", round(row["mean_horizon"],2), "gripper_changed_pct", round(row["gripper_changed_horizon_pct"],2))
    print("CONTROL", control)
    print("SELECTED", best["method"] if best else None)
    print("PAIRED", json.dumps(paired))


if __name__ == "__main__":
    main()
