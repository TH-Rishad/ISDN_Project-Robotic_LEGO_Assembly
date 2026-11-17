"""Execute one structure in Sakib's Newton cell with Richard's motion adapter.

Stripped-down derivative of Kean/scripts/m05.py: the M0.5 gate, stress jobs,
experimental grasp offsets and the opt-in Newton fit are removed. What remains
is the certified acceptance execution path:

    cell (Sakib dual_arm_sim) + cuRobo full-path planning + joint playback
    + clearance checks + scripted wriggle + generated brace search/hold.

GPU dependencies are lazy: importing this module needs no Newton.
"""


def cycle_s(example):
    steps = example.plan["sequence"]
    place_t = getattr(example, "place_t", {}) or {}
    starts = [place_t[step["brick_id"]] for step in steps if step["brick_id"] in place_t]
    if not starts:
        return []
    done_t = getattr(example, "done_t", None)
    end = done_t if done_t is not None else getattr(example, "sim_time", None)
    if end is None:
        return []
    ends = starts[1:] + [end]
    return [round(float(stop) - float(start), 2) for start, stop in zip(starts, ends)]


def cell_args(job, plan):
    """Namespace the shared cell expects; mirrors the acceptance configuration."""
    import argparse as ap
    return ap.Namespace(
        front=None, side=None, top=None, width=None, depth=None, name=None,
        shape=job["shape"] if job["kind"] == "shape" else None,
        structure=job["shape"] if job["kind"] == "structure" else None,
        bricks=None, strategy=job["strategy"],
        brace_model="grasp_lp", legacy_brace=False, brace_json=None, exe="E0",
        plan=str(plan), spec=None, seed=job["seed"], aim="gt", aim_inject=None,
        look=False, shadow_vision=False, show_cameras=False, manual=False,
        camera_every=10, pace=False, viewer="null", test=False, start_step=None,
        rl_control=False, fixture_hold=False, press_scale=None, press_lateral=None,
        p0_out=None, record_all=None, record_inserts=None, save_captures=None,
        continue_=False, monitor=True, repeat=job["seed"],
        tag="richard_%s_s%d" % (job["shape"], job["seed"]),
    )


def run_job(job, out_dir, max_frames, record=True):
    """Run one job (one structure) and return the evidence row.

    Raises SystemExit on a host without the Newton cell (run on HPC).
    """
    from Richard.shared import add_cell_path
    add_cell_path()
    try:
        import newton
        import dual_arm_sim as D
    except ImportError as exc:
        raise SystemExit("runner: Newton cell is not importable (%s). Run this on HPC." % exc)
    out_dir = __import__("pathlib").Path(out_dir)
    plan = out_dir / "plans" / ("%s_%s.json" % (job["shape"], job["strategy"]))
    if plan.exists():
        plan.unlink()
    # Velocity feed-forward for the seven arm joints (planned playback).
    if not getattr(D.build_arm, "_richard_ff", False):
        orig = D.build_arm

        def build_arm_ff(*a, **k):
            arm = orig(*a, **k)
            arm.joint_target_mode[:7] = [int(newton.JointTargetMode.POSITION_VELOCITY)] * 7
            return arm

        build_arm_ff._richard_ff = True
        D.build_arm = build_arm_ff
    viewer = newton.viewer.ViewerNull(num_frames=10 ** 9)
    args = cell_args(job, plan)
    if record:
        name = "%s_%s_s%d" % (job["shape"], job["strategy"], job["seed"])
        args.record_all = str(out_dir / (name + ".npz"))
        args.p0_out = str(out_dir / "cell_rows.jsonl")
    example = D.Example(viewer, args)

    # Brace hold: freeze the measured joints while supporting.
    from Richard.motion.brace_hold import install as install_brace_hold
    install_brace_hold(example)
    # Support handshake timing: the placer waits for the helper to release.
    from Richard.motion.coordinated import install as install_coordination
    install_coordination(example)
    if record:
        from cell.record import Recorder
        example.recorder = Recorder(example)
    example.log_path = out_dir / "episodes.jsonl"
    from cell.contacts import ContactMonitor
    example.monitor = ContactMonitor(example)
    contact_diagnostics = {}
    original_event = example.monitor._event

    def contact_event(cls, body_a, body_b, label_a, label_b, distance, force, ex, seen):
        original_event(cls, body_a, body_b, label_a, label_b, distance, force, ex, seen)
        if cls not in {"penetration", "unintended"}:
            return
        key = (cls, ex.b_step, body_a, body_b)
        previous = contact_diagnostics.get(key)
        if previous is None or distance < previous["distance_m"]:
            poses = ex.state_0.body_q.numpy()
            contact_diagnostics[key] = dict(
                event=cls, step=ex.b_step, frame=ex.frame, distance_m=float(distance),
                a=label_a, b=label_b, force_n=float(force),
                pose_a=poses[body_a].tolist() if body_a >= 0 else None,
                pose_b=poses[body_b].tolist() if body_b >= 0 else None)

    example.monitor._event = contact_event

    from Richard.motion.plan_curobo import CuroboBackend, Planner
    from Richard.motion.tick import Tick
    from Richard.motion.world import plate_box, zone_of
    error = None
    tick = None
    try:
        with Planner(CuroboBackend()) as planner:
            planner.warmup()
            origin = example.plan["voxel_origin"]
            ni, nj = example.NIJ
            pitch = float(example.plan["grid_pitch"][0])

            def world(ex, arm_name):
                from Richard.motion.world import brick_boxes, held_brick
                body_q = ex.state_0.body_q.numpy()
                step = None if ex.b_step is None else ex.plan["sequence"][ex.b_step]
                skip = held_brick(arm_name, ex.B.phase if arm_name == "B" else ex.A.phase, None if step is None else step["brick_id"])
                boxes = [planner.backend.probe.TABLE, plate_box(origin, ni, nj, pitch, 0.0032)]
                boxes.extend(brick_boxes(body_q, ex.body, ex.dims, skip, box_of=planner.backend.probe.brick_box))
                other = 0 if arm_name == "B" else 1
                return boxes, ex.state_0.joint_q.numpy()[9 * other:9 * other + 9]

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

            from Richard.motion.clearance import Clearance
            clearance = Clearance(planner.backend.probe, world)
            from Richard.motion.tick import resolve_wrap
            tick = Tick(planner, zone=zone_of(origin, ni, nj, pitch), world=world,
                        clearance=clearance, wrap_angle=resolve_wrap(D),
                        resample_trajectory=planner.backend.probe.resample,
                        strict=True, carried=carried).attach(example)
            # Tight-socket insertion (the arch crown) and generated brace search.
            from Richard.motion.wriggle import install as install_wriggle
            install_wriggle(example, D)
            from Richard.motion.brace_search import install as install_brace_search
            install_brace_search(example, planner, world, carried, D)

            frames = 0
            while (not example.done or not (example.A.idle() and example.B.idle())) \
                    and example.failure is None and frames < max_frames:
                example.step()
                frames += 1
    except BaseException as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
        raise
    finally:
        example.finish(error=error)
    events = example.monitor.events if example.monitor else []
    row = {
        "shape": job["shape"], "seed": job["seed"], "strategy": job["strategy"],
        "placed": sum(e["success"] for e in example.episodes), "total": len(example.plan["sequence"]),
        "failure": example.failure, "frames": example.frame,
        "events": events,
        "contact_diagnostics": list(contact_diagnostics.values()),
        "brace_search": getattr(example, "brace_search_report", []),
        "cycle_s": cycle_s(example),
        "min_clearance_m": None if tick is None else tick.report().get("min_clearance_m"),
        "motion": "coordinated",
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
