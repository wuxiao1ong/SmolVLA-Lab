from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from analyze_recent import wilson

root = Path.home() / "vla_workspace/vla_interview_project/outputs_recent"
chosen = json.loads((root / "selected_confirmation.json").read_text())["method"]
summaries = []
outcomes = {}
episode_rows=[]
for method in ("base10", chosen):
    successes = []
    calls = replans = 0
    sampling_s = eval_s = 0.0
    per_task = []
    for task in (0,1):
        folder = root / "confirmation" / method / f"libero_spatial_task{task}_seed4000"
        manifest = json.loads((folder / "manifest.json").read_text())
        exploratory = json.loads((root / "heldout" / method / f"libero_spatial_task{task}_seed3000/manifest.json").read_text())
        assert manifest["code_sha256"] == exploratory["code_sha256"]
        assert manifest["model_revision"] == exploratory["model_revision"]
        data = json.loads((folder / "eval_info.json").read_text())
        result = data["per_task"][0]["metrics"]["successes"]
        assert len(result) == 10 and manifest["seed"] == 4000
        states=[json.loads(line)["init_state_index"] for line in (folder/"initial_states.jsonl").read_text().splitlines()]
        assert states == list(range(10,20))
        assert manifest["state_offset"] == 10
        chunks = [json.loads(line) for line in (folder / "planner.jsonl").read_text().splitlines() if json.loads(line).get("event") == "chunk"]
        groups=defaultdict(list)
        for chunk in chunks:
            groups[chunk["rollout"]].append(chunk)
        assert len(groups)==10
        for episode,(success,state,group) in enumerate(zip(result,states,groups.values())):
            episode_rows.append(dict(method=method,task=task,episode=episode,seed=4000+episode,init_state_index=state,
                                     success=int(success),model_calls=sum(c["model_calls"] for c in group),
                                     video=data["overall"]["video_paths"][episode]))
        outcomes[(method,task)] = np.asarray(result,dtype=int)
        per_task.append(sum(result)); successes.extend(result)
        calls += sum(c["model_calls"] for c in chunks)
        replans += len(chunks)
        sampling_s += sum(c["sampling_s"] for c in chunks)
        eval_s += data["overall"]["eval_s"]
    summaries.append(dict(method=method, successes=sum(successes), episodes=len(successes), task_successes=per_task,
                          wilson95=wilson(sum(successes),len(successes)), model_calls=calls,
                          calls_per_episode=calls/len(successes), replans_per_episode=replans/len(successes),
                          sampling_s_per_episode=sampling_s/len(successes), eval_s_per_episode=eval_s/len(successes)))
rng=np.random.default_rng(42)
boot=np.zeros(5000)
differences=[]
for task in (0,1):
    diff=outcomes[(chosen,task)]-outcomes[("base10",task)]
    differences.extend(diff)
    boot+=rng.choice(diff,(5000,10),replace=True).sum(1)
gain=100*float(np.mean(differences))
interval=(100*np.quantile(boot/20,[.025,.975])).tolist()
base_only=sum(int(x==-1) for x in differences)
modified_only=sum(int(x==1) for x in differences)
result=dict(selected=chosen, seed=4000, summary=summaries, gain_pp=gain, paired_bootstrap95_pp=interval,
            base_only=base_only,modified_only=modified_only)
(root/"confirmation_analysis.json").write_text(json.dumps(result,indent=2))
with (root/"confirmation_episodes.csv").open("w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(episode_rows[0]))
    writer.writeheader();writer.writerows(episode_rows)
lines=["# 独立 seed 与初始状态复核", "", f"方法 `{chosen}` 按 seed 3000 结果与预先保存的选择规则挑选。复核未改动模块参数、模型和任务，使用新 seed 4000 和初始状态索引 10–19。每项任务 10 episode，保留该组结果独立于探索结果。", "",
       "源码核查发现 init_states=true 时，seed 不改变初始状态索引。确认前已为两组共同加入偏移 10，并通过 worker 写出的 initial_states.jsonl 验证实际使用索引恰为 10–19；主实验使用默认索引 0–9。这是确认阶段的共享评估配置修正。", "",
       "| 方法 | task 0 | task 1 | 成功数 | 实际模型采样/episode | 同步采样秒/episode |", "|---|---:|---:|---:|---:|---:|"]
for s in summaries:
    lines.append(f"| {s['method']} | {s['task_successes'][0]}/10 | {s['task_successes'][1]}/10 | {s['successes']}/20 | {s['calls_per_episode']:.1f} | {s['sampling_s_per_episode']:.2f} |")
lines += ["", f"独立复核差值：{gain:+.1f} 个百分点，探索性配对 bootstrap 95% 区间 {interval[0]:.1f} 到 {interval[1]:.1f}。基线独有成功/修改独有成功为 {base_only}/{modified_only}。", ""]
if gain>0:
    lines.append("复核中观察到同方向成功率改善，但小样本和区间限制仍需保留；这只是这两个任务上的证据。")
elif gain==0:
    lines.append("探索阶段的成功率改善没有在独立复核中重现；成功数相同，不能声称模块稳定优于 fixed10。")
else:
    lines.append("独立复核表现低于基线；探索阶段的改善没有重现，不能推荐该模块为性能提升方案。")
lines += ["", f"独立复核实际模型采样开销比为 {summaries[1]['model_calls']/summaries[0]['model_calls']:.2f} 倍。成功率与代价必须一起判断。", "",
          "两个任务都涉及把黑碗放到盘子，空间关系与初始状态不同；这里的结论不覆盖其他物体、开关/抽屉操作或真实机器人。", "",
          "## 成功与新增失败案例", "",
          "复核 task 1：episode 0、2（实际初始状态 10、12）由基线失败变为重排成功；episode 7（状态 17）由基线成功变为重排失败。",
          "已查看 episode 0：基线走到 280 步，重排约 118 帧触发成功。episode 7 中基线约 108 帧成功；重排在盘边持续调整至步数上限。视频支持行为差异，但不足以确定是哪一个评分项造成；历史一致性保留了错误路径是待验证机制假设。", "",
          "![Recovered failure on reserved state 10](confirmation_failure_task1_episode0.png)", "",
          "![New failure on reserved state 17](confirmation_failure_task1_episode7.png)"]
(root/"CONFIRMATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
main=(root/"RESULTS.md").read_text()
opening=["完成 160 个正式比较 episode（120 个主试验 + 40 个保留初始状态复核），另有 1 个真实 Smoke Test。修改的是冻结 SmolVLA 的动作候选选择与执行调度模块，模型权重固定。", "",
         "| 阶段 | fixed10 | 候选重排 | 差值 |", "|---|---:|---:|---:|",
         "| 主试验：状态 0–9，seed 3000 | 18/20 | 20/20 | +10 个百分点 |",
         f"| 独立复核：状态 10–19，seed 4000 | {summaries[0]['successes']}/20 | {summaries[1]['successes']}/20 | {gain:+.0f} 个百分点 |", "",
         f"复核差值的配对区间为 {interval[0]:.0f} 到 {interval[1]:.0f} 个百分点，仍含 0。观察到改进方向，但不能宣称已证明稳定优势。代价是每 episode 模型采样从 {summaries[0]['calls_per_episode']:.2f} 增到 {summaries[1]['calls_per_episode']:.2f}（{summaries[1]['model_calls']/summaries[0]['model_calls']:.2f} 倍）；复核恢复 {modified_only} 个失败并引入 {base_only} 个新失败。", "",
         "本轮最值得继续验证的是候选选择模块。AAC 或夹爪熵没有带来超出单独候选选择的已观察成功率收益；对计算预算敏感的使用场景，fixed10 仍是应保留的强基线。更复杂模块需要用增益和代价证明必要性。", "", "---", ""]
(root/"FINAL_REPORT.md").write_text("\n".join(opening)+main+"\n---\n\n"+"\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
