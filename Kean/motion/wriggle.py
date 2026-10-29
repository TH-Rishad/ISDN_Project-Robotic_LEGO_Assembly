"""Wire Tanvir's scripted wriggle into the Newton cell for precision sockets.

The cell's insert is a single straight-down waypoint. A brick dropped into a
SLOT / POCKET / ENCLOSED socket has ~0.1 mm of play per side
(Tanvir/tight_fit_audit.py), so on aim alone it drifts and jams on a flush side
neighbour (the arch crown does exactly this). Replay ScriptedBase's search
phases -- descend, wiggle, press (Tanvir/tasks/insertion_env.py, WRIGGLE_FIT_RL
§3) -- as waypoints on arm B's queue for those steps.

The twin's ScriptedWriggle runs an impedance hand and cannot be imported here
(it binds MuJoCo). This adapter reuses the phase logic and constants through the
cell's own waypoint queue; the classification rule is reused from
tight_fit_audit (re-expressed, not a module copy). The planner's vertical-press
brace (weakest_joint) is untouched; `wriggle_brace.choose` for a sideways
neighbour is a later refinement, not part of this first wiring.
"""

import math
import os
from types import MethodType

import numpy as np

# ScriptedBase search (Tanvir/tasks/insertion_env.py §WP4.4): 2 mm spiral,
# +-1.5 deg wiggle @ 2 Hz, 8 N down. The waypoint queue replays the lateral and
# tilt part; the press is the cell's own final descent.
WIGGLE_AMP = 0.002                # m lateral half-amplitude along the slot axis
WIGGLE_TILT = math.radians(1.5)   # rad rocking about the axis across the slot
WIGGLE_HZ = 2.0                   # 1/s
WIGGLE_PERIODS = 4                # full periods of the wiggle
WIGGLE_WAYPOINTS = 4 * WIGGLE_PERIODS   # quarter-period samples (0.125 s each at 2 Hz)
APPROACH_CLEAR = 0.012            # m above the seated hand target where the wiggle runs
APPROACH_DUR = 0.5                # s: slow descent to the top of the wiggle band
PRESS_DUR = 0.6                   # s: final press, level


def _tuples(plan):
    """Planner tuples (id, type, i, j, k, yaw) in build order."""
    by = {b["id"]: b for b in plan["bricks"]}
    return [(s["brick_id"], by[s["brick_id"]]["type"], *by[s["brick_id"]]["grid_pos"],
             by[s["brick_id"]]["yaw_index"]) for s in plan["sequence"]]


def _side_cells(brick, P):
    _, btype, i0, j0, _, yaw = brick
    nx, ny = P.footprint(btype, yaw)
    return {"-x": [(i0 - 1, j0 + t) for t in range(ny)],
            "+x": [(i0 + nx, j0 + t) for t in range(ny)],
            "-y": [(i0 + t, j0 - 1) for t in range(nx)],
            "+y": [(i0 + t, j0 + ny) for t in range(nx)]}


def contact_faces(brick, placed, P):
    """Faces of `brick` touched by a placed brick of its own layer
    (the tight_fit_audit.contact_faces rule)."""
    k = brick[4]
    same = set().union(*[P.cells(p)[0] for p in placed if p[4] == k]) if placed else set()
    return {s for s, cs in _side_cells(brick, P).items() if any(c in same for c in cs)}


def _critical(contact):
    """SLOT / POCKET / ENCLOSED (tight_fit_audit.classify -> critical)."""
    n = len(contact)
    slot = {"-x", "+x"} <= contact or {"-y", "+y"} <= contact
    return n >= 3 or (n == 2 and slot)


def slot_axes(contact):
    """Horizontal axes along which the socket clamps the brick (the tight fit)."""
    return [ax for ax, faces in (("x", {"-x", "+x"}), ("y", {"-y", "+y"})) if faces <= contact]


def needs_wriggle(ex, step, P):
    """True when the brick drops into a critical (precision) socket."""
    order = _tuples(ex.plan)
    n = step["step"]
    return _critical(contact_faces(order[n], order[:n], P))


def _wiggle_unit(axes):
    """Unit lateral direction(s) the wiggle sweeps: the clamped slot axis/axes."""
    u = np.zeros(3)
    for ax in axes:
        u += np.array([1.0, 0.0, 0.0]) if ax == "x" else np.array([0.0, 1.0, 0.0])
    n = np.linalg.norm(u)
    return u / n if n > 0 else np.array([1.0, 0.0, 0.0])


def _tilt_axis(axes):
    """Tilt rocking axis: the horizontal axis across the slot (y-slot rocks about x)."""
    if axes == ["y"]:
        return np.array([1.0, 0.0, 0.0])
    if axes == ["x"]:
        return np.array([0.0, 1.0, 0.0])
    return np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)


def _move(phase, pos, yaw, grip, dur, tilt=(0.0, 0.0, 0.0), then=None):
    """A waypoint matching Arm.push's move dict (contact moves always 'arrive')."""
    return dict(phase=phase, pos=np.array(pos, float), yaw=yaw, grip=grip, dur=dur,
                wait=None, then=then, contact=True, tilt=np.array(tilt, float), rel=None)


def _splice(ex, step, P):
    """Replace the single insert waypoint with a descend -> wiggle -> press band."""
    B = ex.B
    idx = None
    for i, m in enumerate(B.moves):
        if m["phase"] == "insert":
            idx = i
            break
    if idx is None:
        return
    insert = B.moves[idx]
    bid = step["brick_id"]
    final = np.asarray(insert["pos"], float)          # tgt + [0,0,GRASP_DZ-PRESS]
    yaw, grip, snap = insert["yaw"], insert["grip"], insert["then"]

    order = _tuples(ex.plan)
    n = step["step"]
    axes = slot_axes(contact_faces(order[n], order[:n], P))
    unit = _wiggle_unit(axes)
    tilt_ax = _tilt_axis(axes)

    band = [_move("insert", final + [0.0, 0.0, APPROACH_CLEAR], yaw, grip, APPROACH_DUR)]
    step_z = APPROACH_CLEAR / WIGGLE_WAYPOINTS
    dt = 1.0 / WIGGLE_HZ / 4.0
    for k in range(WIGGLE_WAYPOINTS):
        phase = 2.0 * math.pi * WIGGLE_HZ * (k * dt)
        off = unit * (WIGGLE_AMP * math.sin(phase))
        tilt = tilt_ax * (WIGGLE_TILT * math.sin(phase))
        pos = final + off + [0.0, 0.0, APPROACH_CLEAR - (k + 1) * step_z]
        band.append(_move("insert", pos, yaw, grip, dt, tilt=tilt))
    band.append(_move("insert", final, yaw, grip, PRESS_DUR, then=snap))

    B.moves[idx:idx + 1] = band


def install(example, D):
    """Wire the wriggle into queue_place for critical sockets. D = dual_arm_sim."""
    if os.environ.get("KEAN_WRIGGLE", "1") == "0":
        return
    P = D.P
    original = example.queue_place

    def queue_place(ex, step):
        original(step)
        if needs_wriggle(ex, step, P):
            _splice(ex, step, P)

    example.queue_place = MethodType(queue_place, example)
