"""Replay saved failed planning requests on HPC without rebuilding the structure."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Kean.motion.plan_curobo import CuroboBackend, Planner, Request


def replay_guard(probe, failure, repeats=3):
    """Recompute a recorded command guard with no concurrent GPU callers."""
    from Kean.motion.clearance import Clearance
    checker = Clearance(probe, None)
    measured = np.asarray(failure['measured_q'], float)
    target = np.asarray(failure['target_q'], float)
    if measured.shape != (18,) or target.shape != (18,) or not np.isfinite([measured, target]).all():
        raise ValueError('guard snapshot needs 18 finite measured and commanded joints')
    n = max(1, int(np.ceil(np.max(np.abs(target - measured)) / .02)))
    gaps = []
    for _ in range(repeats):
        values = []
        for t in np.linspace(0., 1., n + 1):
            q = measured + t * (target - measured)
            values.append(probe.clearance_spheres(checker.spheres('A', q[:9]),
                                                  checker.spheres('B', q[9:])))
        gaps.append(float(np.min(values)))
    return gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path, help='absolute path or path relative to the repository root')
    args = parser.parse_args()
    path = repository_path(args.summary)
    if not path.is_file():
        parser.error('Summary does not exist: %s. Supply the saved GPU-run summary path.' % path)
    payload = json.loads(path.read_text())
    failures = [(run['shape'], f) for run in payload['runs']
                for f in run.get('motion_report', {}).get('failures', [])
                if 'request' in f or {'measured_q', 'target_q'} <= f.keys()]
    if not failures:
        parser.error('No replayable snapshots; use a summary from the updated acceptance runner.')
    failed = False
    with Planner(CuroboBackend()) as planner:
        planner.warmup()
        for shape, failure in failures:
            label = '%s step %s %s' % (shape, failure.get('step'), failure.get('phase'))
            try:
                if 'request' not in failure:
                    gaps = replay_guard(planner.backend.probe, failure)
                    ok = bool(np.isfinite(gaps).all() and min(gaps) >= .001)
                    failed |= not ok
                    print(label, 'GUARD PASS' if ok else 'GUARD FAIL',
                          'clearance mm=', [round(g * 1000, 3) for g in gaps], flush=True)
                    continue
                result = planner.plan(Request(**failure['request']))
                print(label, 'PASS', 'goal=', result['goal'], flush=True)
            except Exception as exc:
                failed = True
                print(label, 'FAIL', str(exc), flush=True)
    raise SystemExit(int(failed))


def repository_path(path):
    """The shared launcher changes cwd; data paths must not depend on it."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


if __name__ == '__main__':
    main()
