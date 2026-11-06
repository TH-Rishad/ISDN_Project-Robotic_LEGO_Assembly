import unittest

from Kean.motion.contact_policy import stud_mating


class StudMatingTest(unittest.TestCase):
    def args(self):
        return dict(phase='insert', lower='A_base', mates=['A_base'], depth=-.0012,
                    stud_type=6, upper_type=8, stud_z=.01035)

    def test_actual_stud_engagement_is_intentional(self):
        self.assertTrue(stud_mating(**self.args()))

    def test_wrong_brick_remains_a_collision(self):
        args = self.args()
        args['lower'] = 'B_other_structure'
        self.assertFalse(stud_mating(**args))

    def test_wall_contact_remains_a_collision(self):
        args = self.args()
        args['stud_type'] = 7
        self.assertFalse(stud_mating(**args))

    def test_excessive_stud_penetration_remains_a_collision(self):
        args = self.args()
        args['depth'] = -.00171
        self.assertFalse(stud_mating(**args))

    def test_transit_contact_is_not_mating(self):
        args = self.args()
        args['phase'] = 'transport'
        self.assertFalse(stud_mating(**args))

    def test_nonstud_cylinder_is_not_exempt(self):
        args = self.args()
        args['stud_z'] = .004
        self.assertFalse(stud_mating(**args))
