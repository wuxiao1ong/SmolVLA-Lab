#!/usr/bin/env python3
"""Inspect SmolVLA module names before choosing PEFT target regex.
Requires the user-trained LeRobot environment; no GPU allocation is requested.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lerobot/smolvla_base')
    parser.add_argument('--out', default='outputs/lora_ablation/module_audit')
    args = parser.parse_args()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model = SmolVLAPolicy.from_pretrained(args.model)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in model.named_modules()]
    (output / 'module_names.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    specs = {
        'B_expert': r'(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))',
        'C_expert_plus_vlm14': r'(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.14\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))',
        'D_expert_plus_vlm13_14': r'(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.(13|14)\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))',
    }
    summary = {}
    for key, regex in specs.items():
        matched = [n for n in names if re.fullmatch(regex, n)]
        expert = [n for n in matched if '.lm_expert.' in n]
        vlm = [n for n in matched if '.vlm.model.text_model.layers.' in n]
        summary[key] = {'target_regex': regex, 'matched': len(matched), 'expert': len(expert), 'vlm': len(vlm), 'modules': matched}
        (output / f'{key}_matched.txt').write_text('\n'.join(matched) + '\n', encoding='utf-8')
        print(f'{key}: matched={len(matched)} expert={len(expert)} vlm={len(vlm)}')
    (output / 'target_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    assert summary['B_expert']['expert'] > 0, 'B did not match Action Expert; inspect module_names.txt'
    assert summary['B_expert']['vlm'] == 0, 'B unexpectedly includes VLM targets'
    assert summary['C_expert_plus_vlm14']['vlm'] > 0, 'C did not match VLM layer 14'
    assert summary['D_expert_plus_vlm13_14']['vlm'] > summary['C_expert_plus_vlm14']['vlm'], 'D did not add VLM layer 13'
    print('PASS: target modules found (static naming check, not training proof)')

if __name__ == '__main__':
    main()
