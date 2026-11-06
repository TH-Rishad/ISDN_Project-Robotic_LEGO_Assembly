"""Identify intentional stud engagement without exempting other collisions."""


def stud_mating(phase, lower, mates, depth, stud_type, upper_type, stud_z,
                brick_height=.0096, stud_height=.0017, inset=.0001):
    expected_z = brick_height + stud_height / 2 - inset
    return (phase in ('insert', 'release') and lower in mates
            and -stud_height <= depth < 0.
            and stud_type == 6 and upper_type in (7, 8)  # Newton cylinder, box, mesh
            and abs(stud_z - expected_z) < 1e-6)
