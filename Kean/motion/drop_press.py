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
PRESS_DWELL = 1.5  # s at most at the seat in a top press; stud_fit ends it once the brick is through


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
    drop = np.asarray(target, float) + [0, 0, grasp_dz - press + release_dz]

    def move(phase, pos, grip, dur, then=None, contact=False):
        return dict(phase=phase, pos=np.array(pos, float), yaw=ref['yaw'], grip=float(grip), dur=dur,
                    wait=None, then=then, contact=contact, tilt=np.array(ref['tilt'], float), rel=None)

    moves[first:end] = [
        move('insert', drop, ref['grip'], 1.2, contact=True),
        move('insert', drop, ref['grip'], DWELL, contact=True),
        move('release', drop, release['grip'], .4, contact=True),
    ] + press_moves(ref, target, release['grip'], snap, release['then'], press=press, tip_below_tcp=tip_below_tcp,
                    brick_h=brick_h, stud_h=stud_h, grasp_offset=grasp_offset)


def top_press(moves, target, *, press, tip_below_tcp, brick_h, stud_h, grasp_offset):
    """Keep the queued insert (and any wriggle) and release; snap after a
    closed-tip press on the brick's top instead of at the insert end
    (--stud-fit: a side grip slips long before a stud's 8.9 N)."""
    end = next(i for i, m in enumerate(moves) if m['phase'] == 'retract')
    last = max(i for i, m in enumerate(moves[:end]) if m['phase'] == 'insert' and m['then'])
    release = next(m for m in moves[last:end] if m['phase'] == 'release')
    snap, moves[last]['then'] = moves[last]['then'], None
    placed, release['then'] = release['then'], None
    moves[end:end] = press_moves(moves[last], target, release['grip'], snap, placed, press=press,
                                 tip_below_tcp=tip_below_tcp, brick_h=brick_h, stud_h=stud_h,
                                 grasp_offset=grasp_offset, dwell=PRESS_DWELL)


def press_moves(ref, target, open_grip, snap, placed, *, press, tip_below_tcp, brick_h, stud_h, grasp_offset, dwell=0.):
    """Rise clear of the brick's studs, close, press its top 1 mm past seated
    with the closed tips (snap), rise (placed)."""
    target = np.asarray(target, float)
    # The hand bias keeps the in-hand grasp offset after release; centre the tips on the brick.
    centre = target - [grasp_offset[0], grasp_offset[1], 0]
    hover = centre + [0, 0, brick_h + 2 * stud_h + RISE_CLEAR + tip_below_tcp]
    seat = centre + [0, 0, brick_h + stud_h - press + tip_below_tcp]

    def move(pos, grip, dur, then=None, contact=False):
        return dict(phase='press', pos=np.array(pos, float), yaw=ref['yaw'], grip=float(grip), dur=dur,
                    wait=None, then=then, contact=contact, tilt=np.array(ref['tilt'], float), rel=None)
    # stud_fit sets the dwell's wait: snap as soon as the brick is through, before
    # the closed tips can squeeze an unwelded 1x1 out (dual_studfit_diag_005).
    tail = [dict(move(seat, 0., .05, then=snap, contact=True), dwell=dwell)] if dwell else []
    return [move(hover, open_grip, .5), move(hover, 0., .4),
            move(seat, 0., PRESS_DUR, then=None if dwell else snap, contact=True), *tail,
            move(hover, 0., .4, then=placed)]

def placement_neighbours(bricks, footprint):
    """{id: {ids}} of bricks in the same or the course below touching by a face or a corner.

    A filled layer seats diagonal bricks within a fraction of a millimetre of
    each other's corners, as close as face neighbours; both are intended
    placement adjacency, not collisions. A seated brick's bottom edge is just
    as close to a lower-course brick beside its supports (dual_pyramid_s3_005
    step 5: a 1x1 0.98 mm from the 1x4 one column over).
    """
    cells = {}
    for b in bricks:
        nx, ny = footprint(b['type'], b['yaw_index'])
        i, j, k = b['grid_pos']
        cells[b['id']] = ({(i + a, j + c) for a in range(nx) for c in range(ny)}, k)
    out = {}
    for bid, (own, k) in cells.items():
        grown = {(x + dx, y + dy) for x, y in own for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        out[bid] = {oid for oid, (other, ok) in cells.items() if oid != bid and ok in (k, k - 1) and grown & other}
    return out
