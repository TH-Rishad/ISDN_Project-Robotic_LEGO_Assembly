"""Play a saved acceptance recording in Viser. Does not step the cell controller."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

KEAN = Path(__file__).resolve().parents[1]
ROOT = KEAN.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Kean.motion.playback import apply_recorded, plan_file, recording_file  # noqa: E402

JOBS = {
    "cube": {"shape": "cube", "kind": "shape", "seed": 0, "strategy": "weakest_joint", "stress": False},
    "arch": {"shape": "arch", "kind": "shape", "seed": 0, "strategy": "weakest_joint", "stress": False},
    "hollow_box": {"shape": "hollow_box", "kind": "shape", "seed": 0, "strategy": "weakest_joint", "stress": False},
    "S3": {"shape": "S3", "kind": "structure", "seed": 0, "strategy": "weakest_joint", "stress": False},
}


def make_viewer(port):
    import newton
    cls = getattr(newton.viewer, "ViewerViser", None)
    if cls is None:
        raise SystemExit("newton.viewer.ViewerViser is missing. Run this on HPC in Sakib's environment.")
    try:
        viewer = cls(port=port)
    except TypeError:
        viewer = cls()
    if hasattr(viewer, "paused"):
        viewer.paused = False
    return viewer


def load_record(path):
    with np.load(path, allow_pickle=True) as z:
        record = {key: z[key] for key in z.files}
    record["brick_ids"] = [str(item) for item in record["brick_ids"]]
    record["phase_names"] = [str(item) for item in record["phase_names"]]
    return record


def show_frame(example, record, index):
    import newton
    state = example.state_0
    joints, bodies = apply_recorded(
        state.joint_q.numpy(), state.body_q.numpy(), record["brick_ids"], example.body,
        record["q_exec"][index], record["brick_q"][index])
    state.joint_q.assign(joints.astype(state.joint_q.numpy().dtype, copy=False))
    newton.eval_fk(example.model, state.joint_q, state.joint_qd, state)
    _, bodies = apply_recorded(
        state.joint_q.numpy(), state.body_q.numpy(), record["brick_ids"], example.body,
        record["q_exec"][index], record["brick_q"][index])
    state.body_q.assign(bodies.astype(state.body_q.numpy().dtype, copy=False))
    step = int(record["b_step"][index])
    example.b_step = None if step < 0 else step
    example.sim_time = float(record["sim_time"][index])
    example.done = False
    names = record["phase_names"]
    example.A.phase = names[int(record["a_phase"][index])]
    example.B.phase = names[int(record["b_phase"][index])]
    if 'own_steps' in record and hasattr(example, 'coordinator'):
        for k, a in enumerate('AB'):
            example.coordinator.builds[a].step = int(record['own_steps'][index, k])
            example.roles[a] = str(record['roles'][index, k])
            example.task_builds[a] = str(record['task_builds'][index, k])
        if hasattr(example, 'replay_events'):
            if index == 0:
                example.replay_event_index = 0
                example.coordinator.pending.clear()
                example.coordinator.active = None
            while example.replay_event_index < len(example.replay_events):
                event = example.replay_events[example.replay_event_index]
                if event['time'] > example.sim_time:
                    break
                arm = event['arm']
                if event['event'] == 'support_requested':
                    example.coordinator.pending[arm] = event['step']
                elif event['event'] == 'support_granted':
                    example.coordinator.pending.pop(arm, None)
                    example.coordinator.active = arm
                elif event['event'] == 'support_released':
                    example.coordinator.active = None
                example.replay_event_index += 1


def paused(viewer):
    check = getattr(viewer, "is_paused", None)
    if callable(check):
        return bool(check())
    return bool(getattr(viewer, "paused", False))


def play(example, record, speed, loop, port=8080):
    viewer = example.viewer
    n = len(record["frame"])
    example.frame_dt = (1.0 / 60.0) / speed
    example.realtime = True
    example.rt_t = time.perf_counter()
    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(example.gui, position="side")
    print("playback: %d frames, %.0f s at 1x. Open http://localhost:%d" % (n, n / 60.0, port), flush=True)
    print("This is the saved recording, not a new simulation. Uncheck Pause if the scene is frozen.", flush=True)
    index = 0
    while getattr(viewer, "is_running", lambda: True)():
        show_frame(example, record, index)
        example.render()
        if paused(viewer) or (index == n - 1 and not loop):
            continue
        index += 1
        if index == n:
            index = 0
            print("playback: looping", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=KEAN / "results" / "instructor_run_001")
    parser.add_argument("--shape", choices=tuple(JOBS), default="arch")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    run = args.run if args.run.is_absolute() else ROOT / args.run
    manifest_file = run / 'manifest.json'
    if manifest_file.is_file() and json.loads(manifest_file.read_text()).get('mode') == 'dual_builder':
        from Kean.dual_build import load_example
        manifest = json.loads(manifest_file.read_text())
        record = load_record(run / manifest['recording'])
        example = load_example(run, make_viewer(args.port))
        if list(example.body) != record['brick_ids']:
            raise SystemExit('recording brick order does not match the dual-build manifest')
        print('Dual playback:', manifest['names'], flush=True)
        play(example, record, args.speed, args.loop, args.port)
        return
    record_path = recording_file(run, args.shape)
    plan = plan_file(run, args.shape)
    record = load_record(record_path)
    from Kean.shared import add_cell_path
    add_cell_path()
    import dual_arm_sim as D
    from Kean.scripts.m05 import _args
    viewer = make_viewer(args.port)
    example = D.Example(viewer, _args(JOBS[args.shape], plan))
    ids = list(example.body)
    if ids != record["brick_ids"]:
        raise SystemExit("recording brick order does not match the saved plan")
    play(example, record, args.speed, args.loop, args.port)


if __name__ == "__main__":
    main()
