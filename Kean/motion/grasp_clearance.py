"""Choose a small pad offset using the shared probe's exact finger boxes.

Only pickup waypoints change. The cell's existing in-hand offset servo brings
that grasp to the original brick target. Unresolved geometry is rejected.
"""
import numpy as np

def choose_offset(gap_at, direction, max_offset=0.002, margin=0.0005):
    """Smallest shift clearing both fingers; no change to contact exemptions."""
    direction = np.asarray(direction, float)
    for distance in np.arange(0.0, max_offset + 1e-9, 0.00025):
        candidates = [distance * direction] if distance == 0 else [distance * direction, -distance * direction]
        for offset in candidates:
            if gap_at(offset) >= margin:
                return offset
    raise RuntimeError('No clear grasp within 2 mm of the planned pad centre')



def grip_candidates(grasp, extent, step=.001):
    """(kind, grasp) in fallback order: the planner's grip, the other axis
    centred, then edge grips nearest first on both axes.

    `extent` is the brick's world (x, y) size in m. Conventions follow
    planner.grasp_for: yaw 90 closes on the +-x faces with the pads sliding
    along y. Edge offsets keep the 17.5 mm pads on a long face; a face shorter
    than a pad stays at least 70 % covered, so a 1x1 or 1x2 can slide its pads
    off a flush in-line neighbour.
    """
    lx, ly = extent

    def make(yaw, off):
        across = lx if yaw == 90 else ly
        return dict(yaw_offset_deg=yaw, grip_width_m=round(across - .0012, 4),
                    tcp_offset_m=[0., off] if yaw == 90 else [off, 0.])
    own = int(grasp['yaw_offset_deg'])
    other = 0 if own == 90 else 90
    edges = []
    for yaw in (own, other):
        along = ly if yaw == 90 else lx
        room = max(0., along / 2 - .0085 - .001, along / 2 + .00875 - .7 * along)
        for off in np.arange(step, room + 1e-9, step):
            edges += [(off, yaw, sign * round(float(off), 4)) for sign in (1, -1)]
    edges.sort(key=lambda e: e[0])
    return ([('planner', grasp), ('other_axis', make(other, 0.))] +
            [('edge', make(yaw, off)) for _, yaw, off in edges])


def install(example, probe, cell):
    original = example.queue_place
    changes = []

    def queue_place(step):
        offset = np.zeros(3)
        if not step['grasp'].get('accessible', True):
            if not all(hasattr(probe, name) for name in ('finger_boxes', 'obb_obb_gap', 'brick_box')):
                raise RuntimeError('shared probe has no finger-box clearance API')
            from scipy.spatial.transform import Rotation
            from Kean.motion.world import brick_boxes
            bid = step['brick_id']
            # Use the actual already-placed neighbour poses, not feeder bricks.
            placed = {s['brick_id']: example.body[s['brick_id']]
                      for s in example.plan['sequence'] if s['step'] in example.placed}
            obstacles = brick_boxes(example.state_0.body_q.numpy(), placed, example.dims, box_of=probe.brick_box)
            yaw = np.deg2rad(step['grasp']['yaw_offset_deg'])
            rotation = Rotation.from_euler('z', yaw).as_matrix() @ Rotation.from_quat(cell.Q_DOWN).as_matrix()
            tcp = np.asarray(example.target[bid][0]) + [0, 0, cell.GRASP_DZ - cell.PRESS]
            width = step['grasp']['grip_width_m'] + 0.0012
            # Check closed and open fingers: the release must clear too.
            openings = (width / 2 - 0.0015, width / 2 + 0.002)

            def gap_at(shift):
                return min((probe.obb_obb_gap(finger, box)
                            for g in openings for finger in probe.finger_boxes(tcp + shift, rotation, g)
                            for box in obstacles), default=float('inf'))

            offset = choose_offset(gap_at, [np.cos(yaw), np.sin(yaw), 0])
            print('grasp offset %s: %s mm' % (bid, np.round(offset * 1000, 3).tolist()), flush=True)
        # Let the shared queue builder construct matching hover/settle closures.
        # Restore the feeder map immediately; only this queued pickup is shifted.
        bid = step['brick_id']
        slot = np.asarray(example.feeder[bid], float)
        example.feeder[bid] = slot + offset
        try:
            original(step)
        finally:
            example.feeder[bid] = slot
        changes.append({'step': step['step'], 'offset_m': offset.tolist()})

    example.queue_place = queue_place
    return changes
