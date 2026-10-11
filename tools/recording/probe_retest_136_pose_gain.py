"""Complete the 2.5/6/10 angular gain sweep with fixed solver and servo."""
import argparse
import json

from diagnose_retest_136_pose import OUT, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("hierarchy", "full"), required=True)
    args = parser.parse_args()
    gain = 6.0 if args.solver == "hierarchy" else 2.5
    rows = []
    for x, y in ((0.7, 0.0), (0.65, 0.3)):
        for enhanced in (False, True):
            label = f"gain_{gain}_{'enhanced' if enhanced else 'original_servo'}"
            rows.append(run(args.solver, label, gain, enhanced, enhanced, x, y))
    (OUT / f"gain_sweep_{args.solver}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
