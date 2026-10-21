"""Run the shared P3 probe; historical Kean extensions are opt-in."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Kean.shared import load_probe


def parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--kean-extensions", action="store_true",
                        help="enable historical experimental P3 modifications")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--step", type=int, action="append", default=[])
    options, _ = parser.parse_known_args(argv)
    # Parse integers as the shared parser does, including --step=02.
    if options.all or 2 in options.step:
        parser.error("Collision-model generation belongs to Sakib; Kean consumes the shared models")
    remaining = [arg for arg in argv if arg != "--kean-extensions"]
    return options, remaining


def main():
    options, remaining = parse_args(sys.argv[1:])
    probe = load_probe()
    if options.kean_extensions:
        from Kean.motion import contact_check, iter_budget, sphere_fit
        iter_budget.install(probe)
        iter_budget.install_latency(probe)
        sphere_fit.install(probe)
        contact_check.install(probe)
    sys.argv = [sys.argv[0], *remaining]
    probe.main()


if __name__ == "__main__":
    main()
