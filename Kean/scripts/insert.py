"""Train the residual inserter from the pre-insert pose. Local path is the kinematic cell."""

import argparse
import sys
from pathlib import Path

KEAN = Path(__file__).resolve().parents[1]
if str(KEAN.parent) not in sys.path:
    sys.path.insert(0, str(KEAN.parent))

from Kean.insertion.cell import KinematicInsert, PreInsertCell  # noqa: E402
from Kean.insertion.env import InsertionEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--studs", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from Kean.insertion.train import train
    cell = PreInsertCell(KinematicInsert(args.studs), at_pre_insert=True)
    out = train(InsertionEnv(cell), samples=args.samples, seed=args.seed, device=args.device or "cpu")
    print("insert: loss %.4f samples %d device %s" % (out["loss"], out["samples"], out["device"]))


if __name__ == "__main__":
    main()
