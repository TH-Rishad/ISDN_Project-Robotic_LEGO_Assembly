"""Tight-fit audit: which placement steps of the repo's assemblies need a precision fit?

For every step of every structure the project already has, look at the brick being
placed and the bricks already placed in its own layer, and classify how enclosed
its socket is:

    FREE      no same-layer neighbour touches the brick's faces
    FLUSH     one face against a neighbour
    CORNER    two adjacent faces (an L-shaped datum, open on the other two)
    SLOT      two opposite faces: the brick drops between two bricks
    POCKET    three faces
    ENCLOSED  all four faces: a hole in the middle of a filled layer

The grid is exact (stud pitch 8.0 mm, footprint 8n - 0.2 mm), so between two placed
neighbours the brick has 0.1 mm of play per side along that axis, against the
cell's 0.8 mm (1 sigma) handoff error (insertion_env.BELIEF_SIGMA). SLOT, POCKET and
ENCLOSED are therefore precision-critical: the brick cannot fall in on aim alone.

Read-only on the rest of the repo: it imports Sakib/brickassembly (planner,
blueprint, brace_bandit) and reads its plan JSONs; it writes only under Tanvir/.

    python tight_fit_audit.py                 # all sources, writes results/
    python tight_fit_audit.py --corbel 0      # skip the generated corbel family

Needs numpy, scipy and pillow (the brace_bandit import pulls in stability.py).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True          # no __pycache__ inside Sakib/
HERE = Path(__file__).resolve().parent
BA = HERE.parent / "Sakib" / "brickassembly"
sys.path.insert(0, str(BA))

import planner as P                     # noqa: E402

OUT = HERE / "results"
CLASSES = ["FREE", "FLUSH", "CORNER", "SLOT", "POCKET", "ENCLOSED"]
CRITICAL = {"SLOT", "POCKET", "ENCLOSED"}
PLAY_MM = 0.1                           # per side between two grid neighbours (8n vs 8n - 0.2 mm)


# --- sources ------------------------------------------------------------------

def _from_plan(path):
    """Brick tuples in executed order from a plan JSON (planner's §2.6 contract)."""
    plan = json.loads(path.read_text())
    by_id = {b["id"]: (b["id"], b["type"], *b["grid_pos"], b["yaw_index"]) for b in plan["bricks"]}
    return [by_id[s["brick_id"]] for s in sorted(plan["sequence"], key=lambda s: s["step"])]


def sources(n_corbel):
    """(name, origin, bricks in build order). Executed plan files are preferred over
    re-sequencing, so the order audited is the order the simulator ran."""
    out = []
    for sid in ("S1", "S2", "S3", "S4", "S5"):
        out.append((sid, "planner.STRUCTURES (benchmark), plans/twin/%s_none.json" % sid,
                    _from_plan(BA / "plans" / "twin" / ("%s_none.json" % sid))))
    for sid in ("S2A", "S3C", "S3L"):
        out.append((sid, "planner.STRUCTURES (v3.0), plans/%s_weakest_joint.json" % sid,
                    _from_plan(BA / "plans" / ("%s_weakest_joint.json" % sid))))
    for name, f in (("cube", "sim_cube_none"), ("hollow_box", "sim_hollow_box_weakest_joint"),
                    ("arch", "sim_arch_none"), ("pyramid", "sim_pyramid_none")):
        origin = "user drawing" if name == "pyramid" else "blueprint sample"
        out.append((name, "%s, plans/%s.json" % (origin, f), _from_plan(BA / "plans" / (f + ".json"))))
    if n_corbel:
        from tasks import brace_bandit as BB
        for s in range(n_corbel):
            out.append(("corbel_%02d" % s, "tasks/brace_bandit.corbel_family(%d)" % s,
                        P.sequence(BB.corbel_family(s))))
    return out


# --- per-step geometry ----------------------------------------------------------

def side_cells(brick):
    """The four rows of grid cells just outside each face of the brick's footprint."""
    _, btype, i0, j0, _, yaw = brick
    nx, ny = P.footprint(btype, yaw)
    return {"-x": [(i0 - 1, j0 + t) for t in range(ny)], "+x": [(i0 + nx, j0 + t) for t in range(ny)],
            "-y": [(i0 + t, j0 - 1) for t in range(nx)], "+y": [(i0 + t, j0 + ny) for t in range(nx)]}


def classify(contact):
    """Socket class from the set of faces in contact."""
    n = len(contact)
    slot_x, slot_y = {"-x", "+x"} <= contact, {"-y", "+y"} <= contact
    if n == 0:
        return "FREE"
    if n == 1:
        return "FLUSH"
    if n == 2:
        return "SLOT" if slot_x or slot_y else "CORNER"
    return "POCKET" if n == 3 else "ENCLOSED"


def above_count(brick, bricks):
    """Bricks of the finished structure resting on this one, directly or through others."""
    on = {b[0]: [] for b in bricks}
    for b in bricks:
        for sid in P.supports(b, bricks):
            on[sid].append(b[0])
    seen, todo = set(), list(on[brick[0]])
    while todo:
        b = todo.pop()
        if b not in seen:
            seen.add(b)
            todo += on[b]
    return len(seen)


def contact_faces(b, placed):
    """Faces of b touched by a placed brick of its own layer, and the touched fraction per face."""
    k = b[4]
    same = set().union(*[P.cells(p)[0] for p in placed if p[4] == k]) if placed else set()
    frac = {s: sum(c in same for c in cs) / len(cs) for s, cs in side_cells(b).items()}
    return {s for s, f in frac.items() if f > 0}, frac


def audit(name, bricks):
    rows = []
    for n, b in enumerate(bricks):
        placed = bricks[:n]
        _, k = P.cells(b)
        contact, frac = contact_faces(b, placed)
        cls = classify(contact)
        # Re-ordering within a layer never changes support (it comes from the layer below), so
        # any brick may go last in its layer: the most enclosed socket this brick can be given.
        last = classify(contact_faces(b, [p for p in bricks if p[4] == k and p is not b])[0])
        nx, ny = P.footprint(b[1], b[5])
        grasp = P.grasp_for(b, placed)
        sup = P.supports(b, placed)
        # a pinch axis is free when neither of its two faces has a neighbour
        free_pinch = [ax for ax, faces in (("x", ("-x", "+x")), ("y", ("-y", "+y")))
                      if not contact & set(faces)]
        slot_axes = [ax for ax, faces in (("x", {"-x", "+x"}), ("y", {"-y", "+y"})) if faces <= contact]
        rows.append({
            "structure": name, "step": n, "brick": b[0], "type": b[1], "layer": k,
            "footprint_studs": [nx, ny], "class": cls, "critical": cls in CRITICAL,
            "class_if_last_in_layer": last,
            "faces_in_contact": sorted(contact),
            "contact_fraction": {s: round(f, 2) for s, f in frac.items()},
            "slot_axes": slot_axes,
            "slot_play_mm": PLAY_MM if slot_axes else None,
            "free_pinch_axes": free_pinch,
            "pinch_blocked": P.pinch_blocked(b, placed, grasp),     # planner's tight-step count
            "mating_studs": sum(sup.values()), "n_supports": len(sup),
            "bricks_above_in_final": above_count(b, bricks),
        })
    return rows


# --- report ---------------------------------------------------------------------

def summarise(rows_by_struct, origins):
    lines = ["# Tight-fit audit", "",
             "Generated by `Tanvir/tight_fit_audit.py` from the plans and generators in `Sakib/brickassembly`.",
             "Classes: FREE, FLUSH (1 face), CORNER (2 adjacent), SLOT (2 opposite), POCKET (3), ENCLOSED (4).",
             "Critical = SLOT/POCKET/ENCLOSED: 0.1 mm play per side on a slot axis.", "",
             "## Per structure", "",
             "| structure | source | bricks | " + " | ".join(CLASSES) + " | critical steps | critical & load-bearing | critical if last in layer |",
             "|---|---|---|" + "---|" * len(CLASSES) + "---|---|---|"]
    total = Counter()
    for name, rows in rows_by_struct.items():
        if name.startswith("corbel_"):
            continue
        c = Counter(r["class"] for r in rows)
        total.update(c)
        crit = [r["step"] for r in rows if r["critical"]]
        crit_lb = [r["step"] for r in rows if r["critical"] and r["bricks_above_in_final"]]
        reord = Counter(r["class_if_last_in_layer"] for r in rows if r["class_if_last_in_layer"] in CRITICAL)
        lines.append("| %s | %s | %d | %s | %s | %s | %s |" % (
            name, origins[name], len(rows), " | ".join(str(c[k]) for k in CLASSES),
            ", ".join(map(str, crit)) or "-", ", ".join(map(str, crit_lb)) or "-",
            ", ".join("%d %s" % (v, k) for k, v in sorted(reord.items())) or "-"))
    corbel = [r for n, rows in rows_by_struct.items() if n.startswith("corbel_") for r in rows]
    if corbel:
        c = Counter(r["class"] for r in corbel)
        n_struct = len({r["structure"] for r in corbel})
        with_crit = len({r["structure"] for r in corbel if r["critical"]})
        reord = Counter(r["class_if_last_in_layer"] for r in corbel if r["class_if_last_in_layer"] in CRITICAL)
        lines.append("| corbel family (%d structures) | tasks/brace_bandit.corbel_family | %d | %s | %d structures with any | - | %s |"
                     % (n_struct, len(corbel), " | ".join(str(c[k]) for k in CLASSES), with_crit,
                        ", ".join("%d %s" % (v, k) for k, v in sorted(reord.items())) or "-"))
    lines += ["", "## Every critical or finger-blocked step (hand-authored, sample and user structures)", "",
              "Finger-blocked = planner.pinch_blocked > 0: a placed neighbour sits under a fingertip, so the",
              "side pinch cannot descend to seat the brick (Sakib/Docs/plan_v4_rl_place.md, fact 2).", "",
              "| structure | step | brick | type | layer | class | slot axes | free pinch axes | pinch_blocked | mating studs | supports | bricks above |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rows in rows_by_struct.items():
        if name.startswith("corbel_"):
            continue
        for r in rows:
            if r["critical"] or r["pinch_blocked"]:
                lines.append("| %s | %d | %s | %s | %d | %s | %s | %s | %d | %d | %d | %d |" % (
                    name, r["step"], r["brick"], r["type"], r["layer"], r["class"],
                    ",".join(r["slot_axes"]), ",".join(r["free_pinch_axes"]) or "none",
                    r["pinch_blocked"], r["mating_studs"], r["n_supports"], r["bricks_above_in_final"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corbel", type=int, default=30, help="corbel_family seeds to audit (0 = none)")
    args = ap.parse_args()
    rows_by_struct, origins = {}, {}
    for name, origin, bricks in sources(args.corbel):
        rows_by_struct[name] = audit(name, bricks)
        origins[name] = origin
    OUT.mkdir(exist_ok=True)
    (OUT / "tight_fit_audit.json").write_text(json.dumps(rows_by_struct, indent=1))
    (OUT / "tight_fit_audit.md").write_text(summarise(rows_by_struct, origins))
    print((OUT / "tight_fit_audit.md").read_text())


if __name__ == "__main__":
    main()
