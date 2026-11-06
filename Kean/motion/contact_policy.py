"""Identify intentional stud engagement without exempting other collisions."""
import math


def stud_mating(phase, lower, mates, depth, stud_type, upper_type, stud_z,
                brick_height=.0096, stud_height=.0017, inset=.0001,
                contact_local=None, dims=None):
    expected_z = brick_height + stud_height / 2 - inset
    if not (phase in ('insert', 'release', 'press') and lower in mates
            and -stud_height <= depth < 0. and upper_type in (7, 8)):
        return False
    if stud_type == 6:  # Newton cylinder proxy at the calibrated stud height
        return abs(stud_z - expected_z) < 1e-6
    if stud_type != 8 or contact_local is None or dims is None:
        return False
    # A full brick mesh also contains its studs. Accept only a contact above
    # the shell, close to a known stud centre; shell/wall contacts stay unsafe.
    x, y, z = contact_local
    if not brick_height + .0002 <= z <= brick_height + stud_height + .0002:
        return False
    length, width = dims
    centres = [((i + .5) * .008 - length / 2, (j + .5) * .008 - width / 2)
               for i in range(round(length / .008)) for j in range(round(width / .008))]
    return min((math.hypot(x - cx, y - cy) for cx, cy in centres), default=math.inf) <= .0027
