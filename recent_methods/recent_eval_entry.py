import argparse
import sys


def main():
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--gripper-boundary", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--state-trace")
    args = parser.parse_args(sys.argv[1:split])
    if args.state_trace:
        from state_audit import install_state_audit
        install_state_audit(offset=args.state_offset, trace=args.state_trace)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION
    from recent_controller import install_controller
    install_controller(SmolVLAPolicy, ACTION, method=args.method, trace=args.trace,
                       gripper_boundary=args.gripper_boundary, seed=args.seed)
    sys.argv = ["lerobot-eval"] + sys.argv[split + 1:]
    from lerobot.scripts.lerobot_eval import main as run
    run()


if __name__ == "__main__":
    main()
