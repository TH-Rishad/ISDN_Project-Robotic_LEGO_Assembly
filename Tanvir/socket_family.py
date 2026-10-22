"""Socket contexts for the wriggle-fit policy (WRIGGLE_FIT_RL.md §4).

A context is one placement: the bricks already standing (with their supports) and the
target brick that goes into a socket among them, in the planner's plan format
(`bricks` with grid_pos / yaw_index, `sequence` with mating_studs), so
sim/mj/scene.build_cell takes it unchanged.

Two sources:
  * family(seed, cls): procedural sockets of a requested class (FREE ... ENCLOSED, the
    tight_fit_audit classes) on layer 0-2, with neighbours of random type and overlap,
    support layers tiled by blueprint.tile, and on some draws a neighbour left on few
    studs (a fragile, crucial-layer neighbour). Deterministic in seed.
  * repo_contexts(): the tight steps of the repo's real assemblies (executed SLOTs,
    re-ordered SLOT/POCKETs, the pyramid's finger-blocked steps): the held-out test set.

    python socket_family.py              # class counts, a few samples, writes results/socket_family.json
"""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                       # noqa: E402

import tight_fit_audit as TA             # noqa: E402  (puts Sakib/brickassembly on sys.path)
import blueprint as BP                   # noqa: E402
import planner as P                      # noqa: E402

CLASSES = TA.CLASSES
FACES = ("-x", "+x", "-y", "+y")
FACE_SETS = {
    "FREE": [()],
    "FLUSH": [(f,) for f in FACES],
    "CORNER": [(a, b) for a in ("-x", "+x") for b in ("-y", "+y")],
    "SLOT": [("-x", "+x"), ("-y", "+y")],
    "POCKET": [tuple(f for f in FACES if f != g) for g in FACES],
    "ENCLOSED": [FACES],
}
TARGET_TYPES = {"2x4": 0.3, "2x2": 0.2, "2x3": 0.1, "2x6": 0.1, "1x2": 0.1, "1x4": 0.1, "1x6": 0.05, "1x1": 0.05}
NEIGHBOUR_TYPES = ["1x2", "1x4", "2x2", "2x3", "2x4"]
LAYER_P = (0.5, 0.35, 0.15)              # target on layer 0 / 1 / 2
FRAGILE_P = 0.5                          # draws that leave one neighbour on 1-2 studs
SPLITS = {"train": 0, "val": 100_000, "test": 200_000}   # seed offsets: disjoint by construction


# --- geometry helpers ------------------------------------------------------------------

def _cells(t):
    """Grid cells of a planner tuple (id, type, i, j, k, yaw)."""
    return P.cells(t)[0]


def _face_band(face, i0, j0, nx, ny, a, b, rng):
    """(i, j) corner of an a x b neighbour flush against `face` of the target, overlapping it
    by at least one cell along the face (random overlap)."""
    if face == "-x":
        return i0 - a, int(rng.integers(j0 - b + 1, j0 + ny))
    if face == "+x":
        return i0 + nx, int(rng.integers(j0 - b + 1, j0 + ny))
    if face == "-y":
        return int(rng.integers(i0 - a + 1, i0 + nx)), j0 - b
    return int(rng.integers(i0 - a + 1, i0 + nx)), j0 + ny


def contact_faces(target, placed):
    return TA.contact_faces(target, placed)


def free_pinch_axes(contact):
    return [ax for ax, faces in (("x", ("-x", "+x")), ("y", ("-y", "+y"))) if not contact & set(faces)]


# --- plan-format conversion -------------------------------------------------------------

def to_context(cid, bricks, target_id, source, extra=None):
    """Planner tuples (target last) -> context dict in plan format, grid shifted to start at 0."""
    i_min = min(i for t in bricks for i, _ in _cells(t))
    j_min = min(j for t in bricks for _, j in _cells(t))
    shifted = [(t[0], t[1], t[2] - i_min, t[3] - j_min, t[4], t[5]) for t in bricks]
    order, placed = [], []
    for t in shifted:
        sup = P.supports(t, placed)
        if t[4] > 0 and not sup:
            raise ValueError("%s: %s floats" % (cid, t[0]))
        order.append({"step": len(order), "brick_id": t[0],
                      "mating_studs": [[s, n] for s, n in sorted(sup.items())]})
        placed.append(t)
    tgt = shifted[-1]
    assert tgt[0] == target_id
    contact, frac = contact_faces(tgt, shifted[:-1])
    ctx = {"id": cid, "source": source, "target": target_id,
           "bricks": [{"id": t[0], "type": t[1], "grid_pos": [t[2], t[3], t[4]], "yaw_index": t[5]}
                      for t in shifted],
           "sequence": order,
           "class": TA.classify(contact), "faces": sorted(contact),
           "contact_fraction": {f: round(v, 2) for f, v in frac.items()},
           "free_pinch_axes": free_pinch_axes(contact),
           "layer": tgt[4], "type": tgt[1], "yaw_index": tgt[5],
           "mating_studs": sum(n for _, n in order[-1]["mating_studs"]) if tgt[4] > 0 else 0,
           "planner_brace": None}
    ctx.update(extra or {})
    return ctx


def tuples(ctx):
    """Context -> planner tuples in sequence order (target last)."""
    by = {b["id"]: b for b in ctx["bricks"]}
    return [(s["brick_id"], by[s["brick_id"]]["type"], *by[s["brick_id"]]["grid_pos"],
             by[s["brick_id"]]["yaw_index"]) for s in ctx["sequence"]]


# --- procedural family ---------------------------------------------------------------------

def family(seed, cls, layer=None):
    """One socket of class `cls`; deterministic in (seed, cls, layer)."""
    rng = np.random.default_rng([seed, CLASSES.index(cls), 9 if layer is None else layer])
    types, tp = list(TARGET_TYPES), np.array(list(TARGET_TYPES.values()))
    for _ in range(500):
        ttype = types[rng.choice(len(types), p=tp / tp.sum())]
        yaw = int(rng.integers(2)) if P.BRICKS[ttype][0] != P.BRICKS[ttype][1] else 0
        k = int(rng.choice(3, p=LAYER_P)) if layer is None else layer
        nx, ny = P.footprint(ttype, yaw)
        i0 = j0 = 8
        target = ("target", ttype, i0, j0, k, yaw)
        occupied = set(_cells(target))
        upper = []
        faces = FACE_SETS[cls][rng.integers(len(FACE_SETS[cls]))]
        ok = True
        for f in faces:
            for _try in range(30):
                nt = NEIGHBOUR_TYPES[rng.integers(len(NEIGHBOUR_TYPES))]
                nyaw = int(rng.integers(2))
                a, b = P.footprint(nt, nyaw)
                i, j = _face_band(f, i0, j0, nx, ny, a, b, rng)
                nb = ("nb%d" % len(upper), nt, i, j, k, nyaw)
                c = set(_cells(nb))
                if not c & occupied:
                    upper.append(nb)
                    occupied |= c
                    break
            else:
                ok = False
        if not ok:
            continue
        contact, _ = contact_faces(target, upper)
        if TA.classify(contact) != cls:
            continue                         # a neighbour wrapped a corner onto another face
        lower, fragile = [], None
        if k > 0:
            column = {c for t in upper + [target] for c in _cells(t)}
            top_set = set(column)
            if upper and rng.random() < FRAGILE_P:       # one neighbour keeps only 1-2 supported studs
                nb = upper[rng.integers(len(upper))]
                keep = rng.choice(sorted(_cells(nb)), size=min(len(_cells(nb)), int(rng.integers(1, 3))),
                                  replace=False)
                top_set -= set(_cells(nb)) - {tuple(c) for c in keep}
                fragile = nb[0]
            NI = max(i for i, _ in column) + 1
            NJ = max(j for _, j in column) + 1
            V = np.zeros((NI, NJ, k), bool)
            for kk in range(k):
                for (i, j) in (top_set if kk == k - 1 else column):
                    V[i, j, kk] = True
            try:
                lower = [("L%s" % t[0][2:],) + t[1:] for t in BP.tile(V)]
            except ValueError:
                continue
            if not all(P.supports(t, lower) for t in upper + [target]):
                continue
        try:
            return to_context("fam_%s_%d_%d" % (cls, k, seed), lower + upper + [target], "target",
                              "socket_family.family(%d, %s)" % (seed, cls),
                              {"fragile_neighbour": fragile})
        except ValueError:
            continue
    raise RuntimeError("family(%d, %s): no valid socket in 500 draws" % (seed, cls))


def pool(split, per_class):
    """per_class contexts of every class from the split's seed range."""
    off = SPLITS[split]
    return {cls: [family(off + s, cls) for s in range(per_class)] for cls in CLASSES}


# --- repo contexts (held-out test set) ------------------------------------------------------

def _planner_brace(name, step):
    """Brace target the repo's weakest-joint plan sets for this executed step, if any."""
    files = {"S2": "plans/twin/S2_weakest_joint.json", "S4": "plans/twin/S4_weakest_joint.json",
             "S5": "plans/twin/S5_weakest_joint.json", "arch": "plans/sim_arch_weakest_joint.json",
             "hollow_box": "plans/sim_hollow_box_weakest_joint.json"}
    if name not in files:
        return None
    seq = json.loads((TA.BA / files[name]).read_text())["sequence"]
    s = next((s for s in seq if s["step"] == step), None)
    if not s or not s.get("requires_brace") or not s.get("brace"):
        return None
    return s["brace"].get("target_brick_id")


def repo_contexts():
    """The repo's tight steps: executed critical steps, re-ordered (last in layer) critical
    sockets and finger-blocked steps. Bricks above the target's layer are not yet built."""
    out, seen = [], set()
    for name, _, bricks in TA.sources(0):
        rows = TA.audit(name, bricks)
        for r, b in zip(rows, bricks):
            k = b[4]
            kinds = []
            if r["critical"] or r["pinch_blocked"]:
                kinds.append(("exec", bricks[:r["step"]]))
            if r["class_if_last_in_layer"] in TA.CRITICAL:
                lower = [t for t in bricks if t[4] < k]
                same = [t for t in bricks if t[4] == k and t is not b]
                kinds.append(("last", P.sequence(lower + same) if lower + same else []))
            for kind, placed in kinds:
                key = (name, b[0], frozenset(t[0] for t in placed))
                if key in seen:
                    continue
                seen.add(key)
                tgt = ("target",) + b[1:]
                ctx = to_context("%s_%s_%s" % (name, b[0], kind), list(placed) + [tgt], "target",
                                 "%s step %d (%s)" % (name, r["step"], "as built" if kind == "exec"
                                                      else "placed last in its layer"),
                                 {"structure": name, "step": r["step"], "brick": b[0], "kind": kind,
                                  "pinch_blocked": r["pinch_blocked"] if kind == "exec" else None,
                                  # the plan's brace for this brick's own executed step (for "last", an approximation)
                                  "planner_brace": _planner_brace(name, r["step"])})
                out.append(ctx)
    return out


def main():
    rep = repo_contexts()
    fam = pool("test", 4)
    print("repo contexts: %d" % len(rep))
    for c in rep:
        print("  %-24s %-9s faces %-18s free pinch %-6s layer %d  bricks %2d  planner brace %s" % (
            c["id"], c["class"], ",".join(c["faces"]) or "-", ",".join(c["free_pinch_axes"]) or "none",
            c["layer"], len(c["bricks"]), c["planner_brace"]))
    print("family (test split, 4 per class):")
    for cls, cs in fam.items():
        print("  %-9s layers %s  bricks %s  types %s" % (cls, [c["layer"] for c in cs],
                                                        [len(c["bricks"]) for c in cs], [c["type"] for c in cs]))
    out = HERE / "results" / "socket_family.json"
    out.write_text(json.dumps({"repo": rep, "family_test_sample": fam}, indent=1))
    print("wrote", out.relative_to(HERE))


if __name__ == "__main__":
    main()
