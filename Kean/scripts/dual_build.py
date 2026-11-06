"""Build two structures concurrently with reciprocal support using Kean's controller."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from Kean.dual_build import D, DualExample, make_args, plans_for, feeders_for, pad_for
from Kean.verify_acceptance import capture_provenance, sha256_file


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=D.jsonable) + '\n')


def recorder(example):
    from cell.record import Recorder

    class DualRecorder(Recorder):
        def __init__(self, ex):
            super().__init__(ex)
            self.extra = {k: [] for k in ('own_steps', 'roles', 'task_builds')}

        def step(self, ex):
            super().step(ex)
            self.extra['own_steps'].append([ex.coordinator.builds[a].step for a in 'AB'])
            self.extra['roles'].append([ex.roles[a] for a in 'AB'])
            self.extra['task_builds'].append([ex.task_builds[a] for a in 'AB'])

        def save(self, path, ex):
            super().save(path, ex)
            with np.load(path) as z:
                fields = {k: z[k] for k in z.files}
            fields.update({k: np.asarray(v) for k, v in self.extra.items()})
            np.savez_compressed(path, **fields)
    return DualRecorder(example)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-a', default='arch', choices=sorted(set(D.B.SAMPLES) | set(D.P.STRUCTURES)))
    parser.add_argument('--build-b', default='S3', choices=sorted(set(D.B.SAMPLES) | set(D.P.STRUCTURES)))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--viewer', choices=('null', 'viser', 'gl'), default='null')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--separation', type=float, default=.32)
    parser.add_argument('--max-frames', type=int, default=60000)
    parser.add_argument('--step-limit', type=int, help='diagnostic: execute at most N new steps per builder')
    parser.add_argument('--start-step-a', type=int, default=0, help='diagnostic nominal start; earlier bricks are pre-placed')
    parser.add_argument('--start-step-b', type=int, default=0, help='diagnostic nominal start; earlier bricks are pre-placed')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if not out.is_relative_to(ROOT / 'Kean'):
        parser.error('--out must be inside Kean/')
    if args.max_frames < 1 or (args.step_limit is not None and args.step_limit < 1):
        parser.error('frame/step limits must be positive')
    if out.exists() and any(out.iterdir()):
        parser.error('output exists; choose a fresh directory')
    names = dict(A=args.build_a, B=args.build_b)
    try:
        plans, connections = plans_for(names, args.separation)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    starts = dict(A=args.start_step_a, B=args.start_step_b)
    for a in 'AB':
        if not 0 <= starts[a] <= len(plans[a]['sequence']):
            parser.error('invalid start step for ' + a)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plans').mkdir()
    for a in 'AB':
        write(out / 'plans' / (a + '.json'), plans[a])
    feeders = feeders_for(plans, [pad_for(plans[a]) for a in 'AB'])
    manifest = dict(mode='dual_builder', names=names, separation=args.separation, starts=starts, feeders=feeders,
                    step_limit=args.step_limit, priority='A', plans={a: 'plans/' + a + '.json' for a in 'AB'},
                    recording='recording.npz', contacts='contacts.json', diagnostic=bool(any(starts.values()) or args.step_limit))
    write(out / 'manifest.json', manifest)
    if args.dry_run:
        for a in 'AB':
            print(a, names[a], 'bricks', len(plans[a]['sequence']), 'supports',
                  [s['step'] for s in plans[a]['sequence'] if s['requires_brace']], 'origin', plans[a]['voxel_origin'])
        return
    from Kean.motion.plan_curobo import CuroboBackend, Planner
    from cell.contacts import ContactMonitor
    if args.viewer == 'viser':
        viewer = D.newton.viewer.ViewerViser(port=args.port)
    elif args.viewer == 'gl':
        viewer = D.newton.viewer.ViewerGL()
    else:
        viewer = D.newton.viewer.ViewerNull(num_frames=args.max_frames)
    cell_args = make_args(names)
    cell_args.viewer = args.viewer
    cell_args.record_all = str(out / 'recording.npz')
    provenance = capture_provenance(ROOT, vars(args) | dict(out=str(out), mode='dual_builder'))
    provenance['simulator'] = 'Newton / SolverMuJoCo (GPU)'
    sources = [*ROOT.glob('Kean/*.py'), *ROOT.glob('Kean/motion/*.py'),
               *ROOT.glob('Sakib/brickassembly/cell/*.py'),
               ROOT / 'Sakib/brickassembly/dual_arm_sim.py', Path(__file__).resolve()]
    provenance['source_hashes'] = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(sources)}
    provenance['plan_hashes'] = {a: sha256_file(out / 'plans' / (a + '.json')) for a in 'AB'}
    write(out / 'provenance.json', provenance)
    example = DualExample(viewer, cell_args, plans, connections, out, starts=starts, step_limit=args.step_limit, feeders=feeders)
    example.settle_feeders()
    example.monitor = ContactMonitor(example)
    example.recorder = recorder(example)
    if hasattr(viewer, 'register_ui_callback'):
        viewer.register_ui_callback(example.gui, position='side')
    error = None
    try:
        with Planner(CuroboBackend(world_capacity=len(example.body) + 80)) as planner:
            planner.warmup()
            example.configure(planner)
            while not example.done and not example.failure and example.frame < args.max_frames:
                if args.viewer != 'null' and not viewer.is_running():
                    break
                example.step()
                if args.viewer != 'null':
                    example.render()
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        example.finish(error)
        rows = {}
        for a, view in example.views.items():
            rows[a] = dict(structure=names[a], placed=sum(r['success'] for r in view.episodes), preplaced=starts[a],
                           total=len(plans[a]['sequence']), completed_steps=example.coordinator.builds[a].step,
                           supports=[dict(step=n, **view.brace_row(n)) for n in sorted(view.brace_by_step)],
                           brace_search=view.brace_search_report if hasattr(view, 'brace_search_report') else [])
        contacts = example.monitor.summary()
        failures = [k for k in ('arm_arm', 'unintended', 'penetration', 'displaced') if contacts.get(k, {}).get('count', 0)]
        parallel = example.actual_parallel_frames
        summary = dict(done=example.done, failure=example.failure, error=error, frames=example.frame,
                       builds=rows, contacts=contacts, parallel_frames=parallel, diagnostic=manifest['diagnostic'],
                       motion=example.tick.report() if hasattr(example, 'tick') else {})
        summary['reserved_parallel_frames'] = example.reservations.parallel_frames if hasattr(example, 'reservations') else 0
        summary['initial_settling_xy_m'] = getattr(example, 'initial_settling_xy_m', None)
        summary['ok'] = bool(example.done and not example.failure and not error and not failures and (parallel > 0 or any(starts.values())))
        summary['full_build'] = summary['ok'] and not manifest['diagnostic']
        write(out / 'summary.json', summary)
        from Kean.verify_dual import verify_run
        verification = verify_run(out)
        write(out / 'verification.json', verification)
        summary['ok'] = summary['ok'] and verification['ok']
        summary['full_build'] = summary['ok'] and not manifest['diagnostic']
        write(out / 'summary.json', summary)
        if verification['errors']:
            print('verification:', verification['errors'], flush=True)
        print('dual build:', json.dumps({k: summary[k] for k in ('ok', 'done', 'failure', 'frames', 'parallel_frames')}), flush=True)
    if not summary['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
