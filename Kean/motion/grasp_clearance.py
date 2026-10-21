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
