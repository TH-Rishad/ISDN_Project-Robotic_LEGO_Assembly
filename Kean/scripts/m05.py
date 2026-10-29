"""M0.5 runner. Dry-run lists the gate. A real run needs the Newton cell on HPC."""

import argparse
import json
import sys
from pathlib import Path

KEAN = Path(__file__).resolve().parents[1]
if str(KEAN.parent) not in sys.path:
    sys.path.insert(0, str(KEAN.parent))

from Kean.motion.gate import fold, jobs, placement_count  # noqa: E402

OUT = KEAN / "results" / "m05"


def write_jobs(path, seeds, stress_seeds):
    rows = jobs(seeds, stress_seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "jobs": rows,
        "job_count": len(rows),
        "placements": placement_count(rows),
        "stress_jobs": sum(row["stress"] for row in rows),
        "note": "235 placements: cube, arch, hollow_box, S3 x 5 seeds. Arm-arm is split by braced placement. S3 brace-all x 3 is the zone stress.",
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def _args(job, plan):
    import argparse as ap
    ns = ap.Namespace(
        front=None, side=None, top=None, width=None, depth=None, name=None,
        shape=job["shape"] if job["kind"] == "shape" else None,
        structure=job["shape"] if job["kind"] == "structure" else None,
        bricks=None, strategy="weakest_joint" if job["strategy"] == "brace_all" else job["strategy"],
        brace_model="grasp_lp", legacy_brace=False, brace_json=None, exe="E0",
        plan=str(plan), spec=None, seed=job["seed"], aim="gt", aim_inject=None,
        look=False, shadow_vision=False, show_cameras=False, manual=False,
        camera_every=10, pace=False, viewer="null", test=False, start_step=None,
        rl_control=False, fixture_hold=False, press_scale=None, press_lateral=None,
        p0_out=None, record_all=None, record_inserts=None, save_captures=None,
        continue_=False, monitor=True, repeat=job["seed"], tag="m05_%s_%s_s%d" % (job["shape"], job["strategy"], job["seed"]),
    )
    return ns


def _stress_plan(plan_path, shape):
    from Kean.shared import add_cell_path
    add_cell_path()
    import bracing
    import dual_arm_sim as D
    import planner as P
    import stability as ST
    from Kean.motion.gate import apply_brace_all
    bricks = list(P.STRUCTURES[shape])
    ni = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    nj = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    P.VOXEL_ORIGIN = (-ni * P.PITCH / 2, -nj * P.PITCH / 2, D.Z0)
    plan = P.build_plan(shape, "weakest_joint", brace_model="grasp_lp")
    order = P.sequence(bricks)

    def brace_of(step):
        n = step["step"]
        if n == 0:
            return None
        brick, placed = order[n], list(order[:n])
        return bracing.assign(brick, placed, "weakest_joint", safety_factor=1e9,
                              unbraced=ST.insertion_utilisation(brick, placed))

    apply_brace_all(plan, brace_of)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan))
    return plan_path


def run_job(job, out_dir, max_frames, record=False, grasp_offset=False, fresh_plan=False, planned_motion=True, strict_motion=False):
    from Kean.shared import add_cell_path
    add_cell_path()
    try:
        import newton
        import dual_arm_sim as D
    except ImportError as exc:
        raise SystemExit("m05: Newton cell is not importable (%s). Run this on HPC." % exc)
    plan = out_dir / "plans" / ("%s_%s.json" % (job["shape"], job["strategy"]))
    if fresh_plan and not job["stress"] and plan.exists():
        plan.unlink()
    if job["stress"]:
        _stress_plan(plan, job["shape"])
    if planned_motion and not getattr(D.build_arm, "_kean_ff", False):
        orig = D.build_arm

        def build_arm_ff(*a, **k):
            arm = orig(*a, **k)
            arm.joint_target_mode[:7] = [int(newton.JointTargetMode.POSITION_VELOCITY)] * 7
            return arm

        build_arm_ff._kean_ff = True
        D.build_arm = build_arm_ff
    viewer = newton.viewer.ViewerNull(num_frames=10 ** 9)
    cell_args = _args(job, plan)
    if record:
        name = "%s_%s_s%d" % (job["shape"], job["strategy"], job["seed"])
        cell_args.record_all = str(out_dir / (name + ".npz"))
        cell_args.p0_out = str(out_dir / "cell_rows.jsonl")
    if strict_motion:
        print("motion: serialized planning; full paths, carried-brick checks, full-sphere fallback, stop on failure", flush=True)
    example = D.Example(viewer, cell_args)
    if not planned_motion or strict_motion:
        from Kean.motion.brace_hold import install as install_brace_hold
        install_brace_hold(example)
    if strict_motion:
        from Kean.motion.coordinated import install as install_coordination
        install_coordination(example)
    if record:
        from cell.record import Recorder
        example.recorder = Recorder(example)
    example.log_path = out_dir / "episodes.jsonl"
    from cell.contacts import ContactMonitor
    example.monitor = ContactMonitor(example)
    from Kean.motion.plan_curobo import CuroboBackend, Planner
    from Kean.motion.tick import Tick
    from Kean.motion.world import plate_box, zone_of
    error = None
    tick = None
    grasp_offsets = []
    try:
        from Kean.motion.gate import stamp
        braced_steps = {s["step"] for s in example.plan["sequence"] if s.get("requires_brace")}

        def pump():
            frames = 0
            while (not example.done or (strict_motion and not (example.A.idle() and example.B.idle()))) and example.failure is None and frames < max_frames:
                example.step()
                frames += 1
                step = -1 if example.b_step is None else example.b_step
                stamp(list(example.monitor.raw) + list(example.monitor._active.values()), step, step in braced_steps)

        if not planned_motion:
            pump()
        else:
            with Planner(CuroboBackend()) as planner:
                planner.warmup()
                if grasp_offset:
                    from Kean.motion.grasp_clearance import install as install_grasp_clearance
                    grasp_offsets = install_grasp_clearance(example, planner.backend.probe, D)
                origin = example.plan["voxel_origin"]
                ni, nj = example.NIJ
                pitch = float(example.plan["grid_pitch"][0])

                def world(ex, arm_name):
                    from Kean.motion.world import brick_boxes, held_brick
                    body_q = ex.state_0.body_q.numpy()
                    step = None if ex.b_step is None else ex.plan["sequence"][ex.b_step]
                    skip = held_brick(arm_name, ex.B.phase if arm_name == "B" else ex.A.phase, None if step is None else step["brick_id"])
                    boxes = [planner.backend.probe.TABLE, plate_box(origin, ni, nj, pitch, 0.0032)]
                    boxes.extend(brick_boxes(body_q, ex.body, ex.dims, skip, box_of=planner.backend.probe.brick_box))
                    other = 0 if arm_name == "B" else 1
                    return boxes, ex.state_0.joint_q.numpy()[9 * other:9 * other + (9 if strict_motion else 7)]

                def carried(ex, arm_name):
                    if arm_name != "B" or ex.B.phase not in {"stage", "transport", "pre-insert"}:
                        return None
                    import numpy as np
                    from scipy.spatial.transform import Rotation
                    bid = ex.plan["sequence"][ex.b_step]["brick_id"]
                    poses = ex.state_0.body_q.numpy()
                    tcp = poses[D.EE + ex.n_arm_bodies]
                    brick = poses[ex.body[bid]]
                    rt = Rotation.from_quat(tcp[3:7]).as_matrix()
                    rb = Rotation.from_quat(brick[3:7]).as_matrix()
                    height = float(D.P.BRICK_H + planner.backend.probe.STUD_H)
                    center = rt.T @ (brick[:3] + rb @ np.array([0., 0., height / 2]) - tcp[:3])
                    quat = Rotation.from_matrix(rt.T @ rb).as_quat()
                    length, width = ex.dims[bid]
                    return dict(dims=[float(length), float(width), height], center=center.tolist(),
                                quat_wxyz=quat[[3, 0, 1, 2]].tolist())

                from Kean.motion.clearance import Clearance
                clearance = Clearance(planner.backend.probe, world)
                from Kean.motion.tick import resolve_wrap
                tick = Tick(planner, zone=zone_of(origin, ni, nj, pitch), world=world, clearance=clearance, wrap_angle=resolve_wrap(D), resample_trajectory=planner.backend.probe.resample, strict=strict_motion, carried=carried if strict_motion else None).attach(example)
                if strict_motion:
                    from Kean.motion.brace_search import install as install_brace_search
                    install_brace_search(example, planner, world, carried, D)
                    from Kean.motion.wriggle import install as install_wriggle
                    install_wriggle(example, D)
                pump()
    except BaseException as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        example.finish(error=error)
    events = example.monitor.events if example.monitor else []
    row = {
        "shape": job["shape"], "seed": job["seed"], "strategy": job["strategy"], "stress": job["stress"],
        "placed": sum(e["success"] for e in example.episodes), "total": len(example.plan["sequence"]),
        "failure": example.failure, "frames": example.frame,
        "braced_placements": len(braced_steps),
        "events": events,
        "grasp_offsets": grasp_offsets,
        "brace_search": getattr(example, "brace_search_report", []),
        "motion": "coordinated" if strict_motion else "planned" if planned_motion else "cell",
        "motion_report": {} if tick is None else tick.report(),
        "done": bool(example.done and example.A.idle() and example.B.idle()),
        "breaks": [
            {"step": b.get("step"), "pair": b.get("pair"), "u": b.get("u"),
             "A": b.get("A_phase"), "B": b.get("B_phase")}
            for b in getattr(getattr(example, "breaker", None), "breaks", [])
        ],
        "required_supports": [s["step"] for s in example.plan["sequence"] if s.get("requires_brace")],
        "supports": [
            {"step": n, "executed": bool(info.get("executed")),
             "pose": (example.plan["sequence"][n].get("brace") or {}).get("brace_pose")}
            for n, info in sorted(getattr(example, "brace_by_step", {}).items())
        ],
    }
    return row, {} if tick is None else tick.report()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--stress-seeds", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--job", type=int, default=None, help="run one job index from the dry-run list")
    parser.add_argument("--out", type=Path, default=OUT, help="output directory inside Kean")
    parser.add_argument("--experimental-grasp-offset", action="store_true",
                        help="opt in to unvalidated pickup offsets; default uses shared grasp execution")
    parser.add_argument("--record", action="store_true", help="use the shared per-frame recorder")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(KEAN):
        parser.error("--out must be inside Kean/")
    if args.max_frames <= 0 or args.seeds < 0 or args.stress_seeds < 0:
        parser.error("max-frames must be positive; seed counts must be nonnegative")
    count = len(jobs(args.seeds, args.stress_seeds))
    if count == 0 or (args.job is not None and not 0 <= args.job < count):
        parser.error("job index is outside the selected job list")
    payload = write_jobs(out / "jobs.json", args.seeds, args.stress_seeds)
    print("m05: %d jobs, %d placements, %d stress" % (payload["job_count"], payload["placements"], payload["stress_jobs"]))
    if args.dry_run:
        return
    selected = payload["jobs"] if args.job is None else [payload["jobs"][args.job]]
    runs, ticks = [], []
    for job in selected:
        print("m05: %s %s seed %d" % (job["shape"], job["strategy"], job["seed"]), flush=True)
        row, tick = run_job(job, out, args.max_frames, record=args.record, grasp_offset=args.experimental_grasp_offset)
        runs.append(row)
        ticks.append(tick)
        counts = {}
        for event in row["events"]:
            counts[event["class"]] = counts.get(event["class"], 0) + 1
        print("m05: placed %d/%d events %s" % (row["placed"], row["total"], counts), flush=True)
        for event in row["events"]:
            if event.get("class") != "unintended":
                continue
            print("m05: unintended %s %s phase %s" % (event.get("a"), event.get("b"), event.get("phase")), flush=True)
        with (out / "runs.jsonl").open("a") as log:
            log.write(json.dumps(row, default=str) + "\n")
    summary = fold(runs, {
        "plan_failures": sum(t["plan_failures"] for t in ticks),
        "captures": sum(t["captures"] for t in ticks),
        "min_clearance_m": min((t["min_clearance_m"] for t in ticks if t["min_clearance_m"] is not None), default=None),
    })
    summary["clearance_scope"] = "planned playback, executed joints, full spheres and measured finger openings; contact segments scored by ContactMonitor"
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print("m05: ok=%s %s" % (summary["ok"], ", ".join(summary["reasons"]) or "clean"))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
