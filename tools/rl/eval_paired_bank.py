"""Run the standard PPO evaluator using an isolated held-out motion bank.

All policy, control, scoring and output behavior comes from rl.evaluate.
The additional bank arguments prevent training-bank fallback during evaluation.
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from panda_cable_grasp.rl import evaluate


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paired-bank-dir", type=Path, required=True)
    args, rest = parser.parse_known_args()
    bank_dir = args.paired_bank_dir.resolve()
    if not (bank_dir / "replay_src_shape_nominal_v1.npz").is_file():
        raise FileNotFoundError(bank_dir)
    original_factory = evaluate.make_rl_env

    def factory(**kwargs):
        env = original_factory(**kwargs)
        base = env.base_env
        base.config.replay_bank_dir = str(bank_dir)
        base.config.replay_seed_fallback = "error"
        base._replay_banks.clear()
        base._shape_banks.clear()
        original_reset = env.reset

        def reset(*a, **kw):
            obs, info = original_reset(*a, **kw)
            seed = kw.get("seed")
            if seed is not None:
                if base.config.initial_shape_bank:
                    assert base._initial_shape_source_seed == seed, ("initial bank mismatch", seed, base._initial_shape_source_seed)
                if base.config.replay_bank:
                    assert base._replay_entry_seed == seed, ("replay bank mismatch", seed, base._replay_entry_seed)
            return obs, info

        env.reset = reset
        return env

    evaluate.make_rl_env = factory
    sys.argv = [sys.argv[0], *rest]
    evaluate.main()


if __name__ == "__main__":
    main()
