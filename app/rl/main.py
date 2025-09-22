from __future__ import annotations

import sys
import json

from .train import parse_args as parse_train_args, train_walk_forward
from .eval import parse_args as parse_eval_args, evaluate_oos
from .policy import parse_args as parse_policy_args, RLPolicy


def _cmd_train(argv):
    sys.argv = ["app.rl.train"] + argv
    spec = parse_train_args()
    out = train_walk_forward(spec)
    print(json.dumps(out, indent=2))


def _cmd_eval(argv):
    sys.argv = ["app.rl.eval"] + argv
    args = parse_eval_args()
    out = evaluate_oos(
        symbol=args.symbol,
        tf=args.tf,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        checkpoint=args.checkpoint,
        train_span=args.train_span,
        test_span=args.test_span,
        stride=args.stride,
    )
    print(json.dumps(out, indent=2))


def _cmd_policy(argv):
    sys.argv = ["app.rl.policy"] + argv
    args = parse_policy_args()
    policy = RLPolicy.latest()
    if args.symbol and args.tf:
        out = policy.predict_from_db_latest(args.symbol, args.tf)
    else:
        out = {
            "meta": {
                "algo": policy.meta.algo,
                "version": policy.meta.version,
                "symbol": policy.meta.symbol,
                "tf": policy.meta.tf,
                "checkpoint": policy.meta.best_checkpoint,
            }
        }
    print(json.dumps(out, indent=2))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.rl <train|eval|policy> [options]")
        sys.exit(1)

    cmd, *rest = sys.argv[1:]
    if cmd == "train":
        _cmd_train(rest)
    elif cmd == "eval":
        _cmd_eval(rest)
    elif cmd == "policy":
        _cmd_policy(rest)
    else:
        print(f"Unknown subcommand: {cmd}")
        print("Usage: python -m app.rl <train|eval|policy> [options]")
        sys.exit(1)


if __name__ == "__main__":
    main()
