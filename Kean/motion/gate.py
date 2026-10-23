"""M0.5 job list and the pass rule. neighbour_rub is reported, not counted."""

COUNTS = {"cube": 8, "arch": 11, "hollow_box": 12, "S3": 16}
M0_MIN = {"cube": 8, "arch": 11, "hollow_box": 12, "S3": 15}


def jobs(seeds=5, stress_seeds=3):
    rows = []
    for shape, bricks in COUNTS.items():
        kind = "structure" if shape == "S3" else "shape"
        for seed in range(seeds):
            rows.append({
                "shape": shape, "kind": kind, "seed": seed, "strategy": "weakest_joint",
                "stress": False, "bricks": bricks,
            })
    for seed in range(stress_seeds):
        rows.append({
            "shape": "S3", "kind": "structure", "seed": seed, "strategy": "brace_all",
            "stress": True, "bricks": COUNTS["S3"],
        })
    return rows


def placement_count(rows):
    return sum(row["bricks"] for row in rows if not row["stress"])


def apply_brace_all(plan, brace_of):
    for step in plan["sequence"]:
        if step["step"] == 0:
            step["requires_brace"] = False
            continue
        brace = brace_of(step)
        feasible = bool(brace) and brace.get("feasible", True)
        step["brace"] = brace
        step["requires_brace"] = feasible
    plan.setdefault("validation", {})
    plan["validation"]["brace_required_count"] = sum(bool(s["requires_brace"]) for s in plan["sequence"])
    plan["validation"]["bracing_strategy"] = "brace_all"
    return plan


def count_events(events):
    out = {
        "arm_arm_braced": 0, "arm_arm_unbraced": 0, "unintended": 0,
        "displaced": 0, "penetration": 0, "neighbour_rub": 0,
    }
    for event in events:
        kind = event["class"]
        if kind == "arm_arm":
            out["arm_arm_braced" if event.get("braced") else "arm_arm_unbraced"] += 1
        elif kind in out:
            out[kind] += 1
    return out


def stamp(records, step, braced):
    for rec in records:
        if "braced" not in rec:
            rec["step"] = step
            rec["braced"] = bool(braced)
    return records


def judge(summary):
    reasons = []
    if summary.get("arm_arm_braced", 1) != 0:
        reasons.append("arm_arm_braced")
    if summary.get("arm_arm_unbraced", 1) != 0:
        reasons.append("arm_arm_unbraced")
    placements = int(summary.get("placements") or 0)
    if placements <= 0 or int(summary.get("unintended") or 0) * 50 > placements:
        reasons.append("unintended")
    if summary.get("displaced", 1) != 0:
        reasons.append("displaced")
    if summary.get("penetration", 1) != 0:
        reasons.append("penetration")
    gap = summary.get("min_clearance_m")
    if gap is None or gap <= 0:
        reasons.append("clearance")
    if summary.get("plan_failures", 1) != 0:
        reasons.append("plan_failures")
    if summary.get("captures", 1) != 0:
        reasons.append("captures")
    for run in summary.get("runs") or []:
        if run.get("stress"):
            continue
        if int(run["placed"]) < M0_MIN[run["shape"]]:
            reasons.append("completion:%s:%s" % (run["shape"], run["seed"]))
    return {"ok": not reasons, "reasons": reasons}


def fold(runs, tick):
    events = [event for run in runs if not run.get("stress") for event in run.get("events") or []]
    counted = count_events(events)
    placements = sum(int(run.get("placed") or 0) for run in runs if not run.get("stress"))
    body = {
        "placements": placements,
        "runs": runs,
        "plan_failures": tick.get("plan_failures", 0),
        "captures": tick.get("captures", 0),
        "min_clearance_m": tick.get("min_clearance_m"),
    }
    body.update(counted)
    verdict = judge(body)
    body.update(verdict)
    return body
