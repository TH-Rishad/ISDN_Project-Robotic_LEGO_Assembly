"""Keep B outside the build while A releases and parks after a brace."""
from types import MethodType

import numpy as np

# A's arm reaches the brace through the volume directly above the keystone.
CLEAR_X = 0.16


def _clear_transport(ex):
    away = 1.0 if ex.B.base[0][0] >= ex.A.base[0][0] else -1.0
    for move in ex.B.moves:
        if move['phase'] != 'transport':
            continue
        pos = np.asarray(move['pos'], float).copy()
        pos[0] += away * CLEAR_X
        move['pos'] = pos
        # These are free-space staging poses, never placement/contact targets.
        # A fixed 160 mm offset is only a candidate, not an IK certificate.
        move['alternate_goals'] = [pos + [dx * away, 0., dz]
                                   for dx, dz in ((0., .08), (0., .16), (-.08, .08), (-.16, .16))]
        print('brace transport: hover %.0f mm toward B, clear of A' % (away * CLEAR_X * 1000), flush=True)
        return


def install(example):
    original = example.queue_place

    def queue_place(ex, step):
        original(step)
        if not (step.get('requires_brace') and not ex.legacy
                and step.get('brace', {}).get('feasible', True)):
            return
        # The shared retract announces retracted too early: B is still over
        # the build when A starts moving. Transfer that callback to a planned
        # escape to B's own park, then wait there until A has parked.
        move = ex.B.moves[-1]
        if move['phase'] != 'retract':
            raise RuntimeError('shared placement queue no longer ends in retract')
        released = move['then']
        move['then'] = None
        ex.B.push('clearance', ex.parks[ex.B], move['yaw'], move['grip'], 1.5,
                  then=released)
        ex.B.push('stage', ex.parks[ex.B], move['yaw'], move['grip'], 0.2,
                  wait=lambda: ex.A.idle())
        _clear_transport(ex)

    example.queue_place = MethodType(queue_place, example)
