import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest

from Kean.insertion.cell import KinematicInsert, NotAtPreInsert, OverPress, PreInsertCell, arrived
from Kean.insertion.env import F_INSERT, InsertionEnv


class Arm:
    def __init__(self, phase, cur, nxt):
        self.phase = phase
        self.cur = cur
        self.moves = [] if nxt is None else [{"phase": nxt}]


class InsertCellTest(unittest.TestCase):
    def test_refuses_to_start_before_pre_insert(self):
        with self.assertRaises(NotAtPreInsert):
            PreInsertCell(KinematicInsert(), at_pre_insert=False)
        self.assertTrue(arrived(Arm("pre-insert", None, "insert")))
        self.assertFalse(arrived(Arm("transport", {"phase": "transport"}, "pre-insert")))

    def test_press_cannot_exceed_the_stud_budget(self):
        cell = PreInsertCell(KinematicInsert(4), True)
        env = InsertionEnv(cell)
        _, _, _, info = env.step(env.cell.scripted_action(0) * 0)
        self.assertLessEqual(info["press_n"], 4 * F_INSERT)
        with self.assertRaises(OverPress):
            cell.apply([0, 0, -1, 0, 0, 0, 1], 4 * F_INSERT + 0.1)


if __name__ == "__main__":
    unittest.main()
