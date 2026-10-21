"""Step 8 gate: finger boxes, not finger spheres. Declared grazes are exempt."""

import json
from pathlib import Path


def _pad_key(pad):
    return "%g" % pad


def clearance(row):
    """Arm spheres stay. The finger component is the exact URDF box gap."""
    mins = dict(row["min_clearance_mm"])
    if "finger_box_min_mm" in row:
        mins["finger"] = row["finger_box_min_mm"]
    return mins


def rescore(summ, graze_bricks):
    """Recompute the pad table. `graze_bricks` maps a recording name to brick ids whose grasp is a declared graze."""
    rows = summ.get("rows") or []
    pads = summ["pads_mm"]
    for row in rows:
        graze = row.get("brick") in graze_bricks.get(row["file"], ())
        row["declared_graze"] = graze
        mins = clearance(row)
        row["gate_clearance_mm"] = {k: float(v) for k, v in mins.items()}
        for pad in pads:
            if graze:
                flags = {k: False for k in mins}
                flags["any"] = False
            else:
                flags = {k: bool(v < pad) for k, v in mins.items()}
                flags["any"] = any(flags.values())
            row["flag_pad%s" % _pad_key(pad)] = flags
    scored = [r for r in rows if not r["declared_graze"]]
    for pad in pads:
        key = "flag_pad%s" % _pad_key(pad)
        summ["pad%s" % _pad_key(pad)] = {
            "flagged": sum(r[key]["any"] for r in scored),
            "false_positives": sum(r[key]["any"] and not r["monitor_events"] for r in scored),
            "true_positives": sum(r[key]["any"] and bool(r["monitor_events"]) for r in scored),
            "missed_events": sum((not r[key]["any"]) and bool(r["monitor_events"]) for r in scored),
        }
    clean = [pad for pad in pads if summ["pad%s" % _pad_key(pad)]["false_positives"] == 0 and summ["pad%s" % _pad_key(pad)]["missed_events"] == 0]
    summ["max_padding_without_false_positive_mm"] = max(clean) if clean else None
    cleanseg = [min(r["gate_clearance_mm"].values()) for r in scored if not r["monitor_events"]]
    summ["min_clean_segment_clearance_mm"] = min(cleanseg) if cleanseg else None
    summ["margin_mm"] = (min(cleanseg) - max(clean)) if (cleanseg and clean) else None
    evseg = [min(r["gate_clearance_mm"].values()) for r in scored if r["monitor_events"]]
    summ["max_event_segment_clearance_mm"] = max(evseg) if evseg else None
    summ["declared_graze_segments"] = sum(r["declared_graze"] for r in rows)
    summ["declared_graze_events"] = sum(bool(r["monitor_events"]) for r in rows if r["declared_graze"])
    summ["finger_check"] = "exact URDF boxes; declared graze (grasp accessible=false) exempt"
    summ["ok"] = bool(summ.get("segments")) and bool(clean) and all(r["monitor_events_known"] for r in rows)
    return summ


def _graze_bricks(probe, rows):
    by_file = {}
    cache = {}
    for row in rows:
        matched = probe.TAG_RE.match(Path(row["file"]).stem)
        if not matched:
            by_file.setdefault(row["file"], set())
            continue
        key = (matched["structure"], matched["model"])
        if key not in cache:
            plan, _, _ = probe.world_for_plan(*key)
            cache[key] = {st["brick_id"] for st in plan["sequence"] if not (st.get("grasp") or {}).get("accessible", True)}
        by_file.setdefault(row["file"], cache[key])
    return by_file


def install(probe):
    orig = probe.step8

    def step8(args):
        summ = orig(args)
        if not summ or not summ.get("rows"):
            return summ
        summ = rescore(summ, _graze_bricks(probe, summ["rows"]))
        print("step 8 rescored (finger boxes, declared graze exempt)", flush=True)
        print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1))
        probe.save("step8%s.json" % ("_synthetic" if getattr(args, "synthetic", False) else ""), summ)
        return summ

    probe.step8 = step8
    if getattr(probe, "STEPS", None):
        probe.STEPS[8] = step8
    print("p3: step 8 finger check is exact boxes; declared grazes are exempt", flush=True)
