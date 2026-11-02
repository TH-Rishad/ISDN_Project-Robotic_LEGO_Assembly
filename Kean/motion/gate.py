"""M0.5 job list and the pass rule. neighbour_rub is reported, not counted."""

COUNTS = {"cube": 8, "arch": 11, "hollow_box": 12, "S3": 16}
M0_MIN = {"cube": 8, "arch": 11, "hollow_box": 12, "S3": 15}
P0_CYCLE_S = {"cube": 8.96, "arch": 9.78, "hollow_box": 8.84, "S3": 9.99}
CYCLE_FACTOR = 1.5


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


def mean_cycle(run):
    cycles = [float(value) for value in run.get("cycle_s") or []]
    if not cycles:
        return None
    return sum(cycles) / len(cycles)


def grid_gaps(runs):
    expected = {(row["shape"], row["seed"], row["stress"]) for row in jobs()}
    have = {(run.get("shape"), run.get("seed"), bool(run.get("stress"))) for run in runs}
    return sorted(expected - have)


def judge(summary, full=False):
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
    runs = summary.get("runs") or []
    for run in runs:
        if run.get("stress"):
            continue
        if int(run["placed"]) < M0_MIN[run["shape"]]:
            reasons.append("completion:%s:%s" % (run["shape"], run["seed"]))
    stress = [run for run in runs if run.get("stress")]
    if stress or full:
        if summary.get("stress_arm_arm_braced", 0):
            reasons.append("stress_arm_arm_braced")
        if summary.get("stress_arm_arm_unbraced", 0):
            reasons.append("stress_arm_arm_unbraced")
        if summary.get("stress_displaced", 0):
            reasons.append("stress_displaced")
        if summary.get("stress_penetration", 0):
            reasons.append("stress_penetration")
        stress_placed = int(summary.get("stress_placements") or 0)
        if stress and (stress_placed <= 0 or int(summary.get("stress_unintended") or 0) * 50 > stress_placed):
            reasons.append("stress_unintended")
        for run in stress:
            if int(run.get("placed") or 0) < M0_MIN[run["shape"]]:
                reasons.append("stress_completion:%s:%s" % (run["shape"], run["seed"]))
    if full:
        if grid_gaps(runs):
            reasons.append("grid")
        for run in runs:
            if run.get("stress"):
                continue
            cycle = mean_cycle(run)
            if cycle is None or cycle > CYCLE_FACTOR * P0_CYCLE_S[run["shape"]]:
                reasons.append("cycle:%s:%s" % (run["shape"], run["seed"]))
    return {"ok": not reasons, "reasons": reasons}


def fold(runs, tick, full=False):
    events = [event for run in runs if not run.get("stress") for event in run.get("events") or []]
    counted = count_events(events)
    stress_counted = count_events(event for run in runs if run.get("stress") for event in run.get("events") or [])
    placements = sum(int(run.get("placed") or 0) for run in runs if not run.get("stress"))
    if runs and all("motion_report" in run for run in runs):
        plan_failures = sum(int((run.get("motion_report") or {}).get("plan_failures") or 0) for run in runs)
        captures = sum(int((run.get("motion_report") or {}).get("captures") or 0) for run in runs)
    else:
        plan_failures = tick.get("plan_failures", 0)
        captures = tick.get("captures", 0)
    if runs and all("min_clearance_m" in run for run in runs):
        gaps = [run.get("min_clearance_m") for run in runs]
        gap = None if any(value is None for value in gaps) else min(gaps)
    else:
        gap = tick.get("min_clearance_m")
    body = {
        "placements": placements,
        "stress_placements": sum(int(run.get("placed") or 0) for run in runs if run.get("stress")),
        "runs": runs,
        "plan_failures": plan_failures,
        "captures": captures,
        "min_clearance_m": gap,
        "grid_gaps": grid_gaps(runs),
    }
    body.update(counted)
    body.update({
        "stress_arm_arm_braced": stress_counted["arm_arm_braced"],
        "stress_arm_arm_unbraced": stress_counted["arm_arm_unbraced"],
        "stress_unintended": stress_counted["unintended"],
        "stress_displaced": stress_counted["displaced"],
        "stress_penetration": stress_counted["penetration"],
        "stress_neighbour_rub": stress_counted["neighbour_rub"],
    })
    verdict = judge(body, full=full)
    body.update(verdict)
    return body
