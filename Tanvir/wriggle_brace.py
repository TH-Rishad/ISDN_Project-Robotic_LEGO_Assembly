"""Wriggle-brace trigger (WRIGGLE_FIT_RL.md §3.8): which neighbour should arm A hold while
arm B wriggles a brick into a tight socket?

The planner's brace (bracing.py) answers a different question: the vertical insertion press
levering an overhang. A wriggle loads the neighbours SIDEWAYS: the brick rubs and pushes on
their top edges. For every same-layer neighbour touching the socket, this module puts a
lateral push F_LAT at the neighbour's top edge (lever = one brick height above its stud
joint), directed away from the socket, splits it over the neighbour's support connections by
stud count, and evaluates each connection with the twin's own capacity model
(sim/joint_model/capacity.Patch). The neighbour with the highest utilisation is braced when
it reaches U_TRIGGER.

Calibration: F_LAT is set so the slot probe's pocket (slot_probe.py) flags the 1x2 side
neighbour, the brick that broke in every unbraced pocket episode, and neither 2x2.

    python wriggle_brace.py          # utilisation per neighbour for the probe pocket and the repo contexts
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                         # noqa: E402

import socket_family as SF                                 # noqa: E402
import planner as P                                        # noqa: E402
from sim.joint_model.capacity import F_BREAK_PER_STUD, Patch   # noqa: E402

F_LAT = 8.0          # N, lateral wriggle push on a neighbour's top edge (calibrated, see docstring)
U_TRIGGER = 0.6      # brace when the worst neighbour reaches this utilisation
NORMAL = {"-x": (-1, 0), "+x": (1, 0), "-y": (0, -1), "+y": (0, 1)}   # socket -> neighbour direction


def _patch(cells):
    """Capacity patch of a set of stud cells, in a frame centred on them (scene.build_cell's convention)."""
    xy = np.array([[(i + 0.5) * P.PITCH, (j + 0.5) * P.PITCH] for i, j in cells])
    c = xy.mean(axis=0)
    st = xy - c
    rect = (st[:, 0].min() - P.PITCH / 2, st[:, 0].max() + P.PITCH / 2,
            st[:, 1].min() - P.PITCH / 2, st[:, 1].max() + P.PITCH / 2)
    return Patch(st, rect, F_BREAK_PER_STUD)


def neighbour_utilisation(ctx):
    """{neighbour_id: (utilisation under the wriggle push, face it touches)} for the target's
    same-layer neighbours in a socket_family context."""
    ts = SF.tuples(ctx)
    tgt, placed = ts[-1], ts[:-1]
    by = {t[0]: t for t in placed}
    sides = {f: set(cs) for f, cs in SF.TA.side_cells(tgt).items()}
    out = {}
    for t in placed:
        if t[4] != tgt[4]:
            continue
        cells = SF._cells(t)
        touched = [f for f, cs in sides.items() if cs & cells]
        if not touched:
            continue
        sup = P.supports(t, placed) if t[4] > 0 else {None: len(cells)}
        total = sum(sup.values())
        u = 0.0
        for sid, n in sup.items():
            patch_cells = cells if sid is None else cells & SF._cells(by[sid])
            pa = _patch(patch_cells)
            share = F_LAT * n / total
            for f in touched:
                fx, fy = NORMAL[f]
                for sgn in (1, -1):                         # a wriggle pushes and drags
                    Fx, Fy = sgn * share * fx, sgn * share * fy
                    Mx, My = -P.BRICK_H * Fy, P.BRICK_H * Fx    # r = (0, 0, H) above the joint
                    u = max(u, float(pa.utilization(0.0, Mx, My)))
        out[t[0]] = (round(u, 3), touched)
    return out


def choose(ctx):
    """(neighbour id or None, its utilisation): the wriggle brace for this context."""
    u = neighbour_utilisation(ctx)
    if not u:
        return None, 0.0
    nb, (val, _) = max(u.items(), key=lambda kv: kv[1][0])
    return (nb if val >= U_TRIGGER else None), val


def probe_pocket():
    """The slot probe's pocket (InsertionEnv stage 3 with a 2x4): 2x2 at -y and +y, 1x2 at +x."""
    bricks = [("nb0", "2x2", 2, 0, 0, 0), ("nb1", "2x2", 2, 6, 0, 0), ("nb2", "1x2", 4, 2, 0, 0),
              ("target", "2x4", 2, 2, 0, 0)]
    return SF.to_context("probe_pocket", bricks, "target", "slot_probe pocket")


def main():
    pc = probe_pocket()
    print("probe pocket:", neighbour_utilisation(pc), "-> brace", choose(pc))
    for ctx in SF.repo_contexts():
        nb, u = choose(ctx)
        print("  %-24s %-9s brace %-8s u %.2f   planner brace %s" % (ctx["id"], ctx["class"], nb, u, ctx["planner_brace"]))


if __name__ == "__main__":
    main()
