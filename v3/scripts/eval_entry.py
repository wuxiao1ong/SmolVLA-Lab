"""CLI bridge, e.g. python scripts/eval_entry.py --mode adaptive --short 5 \
  --long 25 --threshold 0.1 --trace /tmp/t.jsonl -- --policy.path=... ...

Arguments after '--' are passed unchanged to LeRobot's eval_main parser.
"""
import argparse
import sys


def main():
    if "--" not in sys.argv:
        raise SystemExit("Usage: eval_entry.py [controller flags] -- [lerobot-eval flags]")
    ix = sys.argv.index("--")
    controller_flags = sys.argv[1:ix]
    native_flags = sys.argv[ix + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "fixed", "adaptive"], required=True)
    ap.add_argument("--short", type=int, default=5)
    ap.add_argument("--long", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--trace", type=str, required=True)
    options = ap.parse_args(controller_flags)
    # Imported after flags parsed so the script may display --help without LeRobot.
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION
    from adaptive_controller import install_controller

    install_controller(SmolVLAPolicy, ACTION, mode=options.mode,
                       short=options.short, long=options.long,
                       threshold=options.threshold, trace=options.trace)
    sys.argv = ["lerobot-eval"] + native_flags
    from lerobot.scripts.lerobot_eval import main as lerobot_main
    lerobot_main()


if __name__ == "__main__":
    main()
