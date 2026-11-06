"""Drop-and-press for tight steps (Sakib/Docs/plan_v4_rl_place.md §2.1, default setting).

A side pinch cannot seat a brick whose pinched faces are flush with placed
neighbours: the fingertips reach the neighbours' studs first (the pyramid's
step 2 stalled 7.8 mm high). Lower the brick only until the fingers still clear
those studs, release it into the gap, close the empty fingers above it, and
press its top with the closed tips through the snap gate.
"""
import numpy as np

DWELL = 0.4        # s at the release height so the brick-aim servo settles
RISE_CLEAR = .003  # tips above the dropped brick's stud tops before closing
PRESS_DUR = 1.0    # s, as the existing closed-tip legacy brace press


def splice(moves, target, release_dz, *, grasp_dz, press, tip_below_tcp, brick_h, stud_h, grasp_offset):
    """Replace the queued insert..release with drop-and-press, in place.

    `release_dz` is the TCP height above the normal insert end where the
    open and closed fingers clear the neighbours. Reuses the queue's own
    snap and placed callbacks so gating and bookkeeping are unchanged.
    """
    first = next(i for i, m in enumerate(moves) if m['phase'] == 'insert')
    end = next(i for i, m in enumerate(moves) if m['phase'] == 'retract')
    tail = moves[first:end]
    snap = next(m['then'] for m in reversed(tail) if m['phase'] == 'insert' and m['then'])
    release = next(m for m in tail if m['phase'] == 'release')
    ref = tail[0]
    yaw, tilt, shut = ref['yaw'], ref['tilt'], ref['grip']
    target = np.asarray(target, float)
    drop = target + [0, 0, grasp_dz - press + release_dz]
    # The hand bias keeps the in-hand grasp offset after release; centre the tips on the brick.
    centre = target - [grasp_offset[0], grasp_offset[1], 0]
    hover = centre + [0, 0, brick_h + 2 * stud_h + RISE_CLEAR + tip_below_tcp]
    seat = centre + [0, 0, brick_h + stud_h - press + tip_below_tcp]

    def move(phase, pos, grip, dur, then=None, contact=False):
        return dict(phase=phase, pos=np.array(pos, float), yaw=yaw, grip=float(grip), dur=dur,
                    wait=None, then=then, contact=contact, tilt=np.array(tilt, float), rel=None)

    moves[first:end] = [
        move('insert', drop, shut, 1.2, contact=True),
        move('insert', drop, shut, DWELL, contact=True),
        move('release', drop, release['grip'], .4, contact=True),
        move('press', hover, release['grip'], .5),
        move('press', hover, 0., .4),
        move('press', seat, 0., PRESS_DUR, then=snap, contact=True),
        move('press', hover, 0., .4, then=release['then']),
    ]


def placement_neighbours(bricks, footprint):
    """{id: {ids}} of same-course bricks touching by a face or a corner.

    A filled layer seats diagonal bricks within a fraction of a millimetre of
    each other's corners, as close as face neighbours; both are intended
    placement adjacency, not collisions.
    """
    cells = {}
    for b in bricks:
        nx, ny = footprint(b['type'], b['yaw_index'])
        i, j, k = b['grid_pos']
        cells[b['id']] = ({(i + a, j + c) for a in range(nx) for c in range(ny)}, k)
    out = {}
    for bid, (own, k) in cells.items():
        grown = {(x + dx, y + dy) for x, y in own for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        out[bid] = {oid for oid, (other, ok) in cells.items() if oid != bid and ok == k and grown & other}
    return out
