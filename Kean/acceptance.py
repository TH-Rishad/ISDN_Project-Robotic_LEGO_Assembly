"""Arch and S3 from a structure id. The cell executes the generated supports."""

from Kean.shared import CELL, add_cell_path

ACCEPTANCE = (("arch", "shape"), ("S3", "structure"))
STRATEGY = "weakest_joint"


def plate_z():
    for line in (CELL / "dual_arm_sim.py").read_text().splitlines():
        if line.startswith("Z0 = "):
            return float(line.split("=", 1)[1].split("#", 1)[0])
    raise RuntimeError("dual_arm_sim.Z0 missing")


def _planner():
    add_cell_path()
    import planner as P
    return P


def bricks_for(name, kind):
    P = _planner()
    if kind == "structure":
        return list(P.STRUCTURES[name])
    if kind != "shape":
        raise ValueError(kind)
    import blueprint as B
    bricks = list(B.load(name))
    P.STRUCTURES[name] = bricks
    return bricks


def generate(name, kind, strategy=STRATEGY):
    """Direct structure id. Calls planner.sequence and planner.build_plan."""
    if strategy == "none":
        raise ValueError("acceptance does not use strategy none")
    P = _planner()
    saved = P.VOXEL_ORIGIN
    bricks = bricks_for(name, kind)
    ni = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    nj = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    try:
        P.VOXEL_ORIGIN = (-ni * P.PITCH / 2, -nj * P.PITCH / 2, plate_z())
        plan = P.build_plan(name, strategy, brace_model="grasp_lp")
        order = [b[0] for b in P.sequence(list(P.STRUCTURES[name]))]
    finally:
        P.VOXEL_ORIGIN = saved
    got = [s["brick_id"] for s in plan["sequence"]]
    if order != got:
        raise RuntimeError("plan order is not planner.sequence")
    if plan["validation"]["bracing_strategy"] == "none":
        raise RuntimeError("build_plan returned strategy none")
    return plan


def steps_of(plan):
    rows = []
    for step in plan["sequence"]:
        brace = step.get("brace") or {}
        rows.append({
            "step": step["step"],
            "brick_id": step["brick_id"],
            "requires_brace": bool(step.get("requires_brace")),
            "support_pose": brace.get("brace_pose") if step.get("requires_brace") else None,
            "support_brick": brace.get("target_brick_id") if step.get("requires_brace") else None,
        })
    return rows


def preview():
    structures = []
    for name, kind in ACCEPTANCE:
        plan = generate(name, kind)
        structures.append({
            "structure": name,
            "kind": kind,
            "strategy": plan["validation"]["bracing_strategy"],
            "bricks": len(plan["sequence"]),
            "brace_required_count": plan["validation"]["brace_required_count"],
            "single_arm_fails": plan["validation"].get("single_arm_fails"),
            "steps": steps_of(plan),
        })
    return {
        "source": "planner.build_plan and bracing.assign",
        "image_input": False,
        "strategy": STRATEGY,
        "structures": structures,
    }


def preview_ok(payload):
    reasons = []
    if payload.get("image_input") or payload.get("strategy") == "none":
        reasons.append("input")
    got = {row["structure"] for row in payload.get("structures") or []}
    if got != {"arch", "S3"}:
        reasons.append("structures")
    for row in payload.get("structures") or []:
        if row["strategy"] == "none" or row["bricks"] < 8:
            reasons.append("structure:%s" % row["structure"])
        braced = [s for s in row["steps"] if s["requires_brace"]]
        if not braced or any(s["support_pose"] is None for s in braced):
            reasons.append("support:%s" % row["structure"])
    return reasons


def supports_held(example):
    out = []
    for n, info in sorted(getattr(example, "brace_by_step", {}).items()):
        step = example.plan["sequence"][n]
        brace = step.get("brace") or {}
        out.append({
            "step": n,
            "brick_id": step["brick_id"],
            "executed": bool(info.get("executed")),
            "pose": brace.get("brace_pose"),
            "target": brace.get("target_brick_id"),
        })
    return out


def judge(runs):
    reasons = []
    if {r["shape"] for r in runs} != {"arch", "S3"}:
        reasons.append("structures")
    for run in runs:
        if run.get("failure") or int(run.get("placed") or 0) != int(run.get("total") or -1):
            reasons.append("incomplete:%s" % run["shape"])
        required = run.get("required_supports") or []
        if not required:
            reasons.append("no_support:%s" % run["shape"])
        held = {h["step"] for h in run.get("supports") or [] if h.get("executed")}
        for n in required:
            if n not in held:
                reasons.append("unheld:%s:%s" % (run["shape"], n))
    return reasons
