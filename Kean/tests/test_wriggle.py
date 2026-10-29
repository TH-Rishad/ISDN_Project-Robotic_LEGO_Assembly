import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from Kean.motion import wriggle
from Kean.tests.test_tick import Ex


class FakeP:
    """The planner geometry wriggle uses (footprint / cells), for CPU tests."""
    BRICKS = {"2x2": (2, 2), "2x6": (2, 6)}

    def footprint(self, btype, yaw_index):
        nx, ny = self.BRICKS[btype]
        return (ny, nx) if yaw_index else (nx, ny)

    def cells(self, brick):
        _, btype, i, j, k, yaw = brick
        nx, ny = self.footprint(btype, yaw)
        return {(i + a, j + b) for a in range(nx) for b in range(ny)}, k


def arch_plan():
    """A two-pier arch with a 2x6 crown dropped between them (SLOT along y)."""
    bricks = [{"id": "b_000", "type": "2x2", "grid_pos": [0, 0, 1], "yaw_index": 0},
              {"id": "b_001", "type": "2x2", "grid_pos": [0, 8, 1], "yaw_index": 0},
              {"id": "b_002", "type": "2x6", "grid_pos": [0, 2, 1], "yaw_index": 0}]
    seq = [{"step": 0, "brick_id": "b_000"},
           {"step": 1, "brick_id": "b_001"},
           {"step": 2, "brick_id": "b_002"}]
    return {"bricks": bricks, "sequence": seq}


class WriggleTest(unittest.TestCase):
    def test_crown_is_a_critical_slot(self):
        ex = SimpleNamespace(plan=arch_plan())
        order = wriggle._tuples(ex.plan)
        contact = wriggle.contact_faces(order[2], order[:2], FakeP())
        self.assertEqual(contact, {"-y", "+y"})
        self.assertTrue(wriggle._critical(contact))
        self.assertEqual(wriggle.slot_axes(contact), ["y"])
        self.assertTrue(wriggle.needs_wriggle(ex, ex.plan["sequence"][2], FakeP()))

    def test_free_and_flush_steps_do_not_wriggle(self):
        ex = SimpleNamespace(plan=arch_plan())
        self.assertFalse(wriggle.needs_wriggle(ex, ex.plan["sequence"][0], FakeP()))
        # a single flush neighbour (only the left pier placed first) is not critical
        one = {"bricks": arch_plan()["bricks"],
               "sequence": [{"step": 0, "brick_id": "b_000"}, {"step": 1, "brick_id": "b_002"}]}
        order = wriggle._tuples(one)
        contact = wriggle.contact_faces(order[1], order[:1], FakeP())
        self.assertEqual(contact, {"-y"})
        self.assertFalse(wriggle._critical(contact))

    def test_real_planner_classifies_the_arch_crown(self):
        from Kean.acceptance import bricks_for, _planner
        P = _planner()
        bricks_for("arch", "shape")
        order = P.sequence(list(P.STRUCTURES["arch"]))
        brick, placed = order[10], order[:10]
        self.assertEqual(brick[0], "b_010")
        contact = wriggle.contact_faces(brick, placed, P)
        self.assertEqual(contact, {"-y", "+y"})
        self.assertTrue(wriggle._critical(contact))

    def test_splice_replaces_insert_with_wriggle_and_keeps_snap(self):
        ex = Ex()
        ex.plan = arch_plan()
        ex.target = {"b_002": (np.array([0.0, 0.0, 0.05]), 0.0, False)}
        snap = lambda: None
        ex.B.push("pre-insert", [0.0, 0.0, 0.08], 0.0, 0.01, 1.0)
        ex.B.push("insert", [0.0, 0.0, 0.06], 0.0, 0.01, 1.0, then=snap)
        wriggle._splice(ex, ex.plan["sequence"][2], FakeP())
        # one approach + WIGGLE_WAYPOINTS wiggle + one press, all still "insert"
        n = wriggle.WIGGLE_WAYPOINTS
        self.assertEqual([m["phase"] for m in ex.B.moves], ["pre-insert"] + ["insert"] * (n + 2))
        self.assertIs(ex.B.moves[-1]["then"], snap)
        # lateral wiggle sweeps the slot axis (y), not the free axis (x)
        ys = [m["pos"][1] for m in ex.B.moves[1:]]
        xs = [m["pos"][0] for m in ex.B.moves[1:]]
        self.assertGreater(max(ys) - min(ys), 1e-4)
        self.assertLess(max(xs) - min(xs), 1e-9)
        # tilt wiggle rocks about x (across the y slot)
        tilts = [m["tilt"] for m in ex.B.moves[1:]]
        self.assertGreater(max(t[0] for t in tilts), 0.0)
        # final press target is unchanged and level
        np.testing.assert_allclose(ex.B.moves[-1]["pos"], [0.0, 0.0, 0.06])
        np.testing.assert_allclose(ex.B.moves[-1]["tilt"], [0.0, 0.0, 0.0])
        # every wriggle waypoint is a contact move so it never hangs on a jam
        self.assertTrue(all(m["contact"] for m in ex.B.moves[1:]))

    def test_install_only_wriggles_critical_sockets(self):
        ex = Ex()
        ex.plan = arch_plan()
        ex.target = {"b_002": (np.array([0.0, 0.0, 0.05]), 0.0, False)}

        def original(step):
            ex.B.push("insert", [0.0, 0.0, 0.06], 0.0, 0.01, 1.0)

        ex.queue_place = original
        wriggle.install(ex, SimpleNamespace(P=FakeP()))
        # the crown (step 2) is critical -> wriggled
        ex.queue_place(ex.plan["sequence"][2])
        self.assertEqual(len([m for m in ex.B.moves if m["phase"] == "insert"]),
                         1 + wriggle.WIGGLE_WAYPOINTS + 1)
        ex.B.moves.clear()
        # the first pier (step 0) is FREE -> untouched
        ex.queue_place(ex.plan["sequence"][0])
        self.assertEqual([m["phase"] for m in ex.B.moves], ["insert"])

    def test_install_can_be_disabled(self):
        ex = Ex()
        ex.plan = arch_plan()
        ex.target = {"b_002": (np.array([0.0, 0.0, 0.05]), 0.0, False)}

        def original(step):
            ex.B.push("insert", [0.0, 0.0, 0.06], 0.0, 0.01, 1.0)

        ex.queue_place = original
        with mock.patch.dict("os.environ", {"KEAN_WRIGGLE": "0"}):
            wriggle.install(ex, SimpleNamespace(P=FakeP()))
        ex.queue_place(ex.plan["sequence"][2])
        self.assertEqual([m["phase"] for m in ex.B.moves], ["insert"])


if __name__ == "__main__":
    unittest.main()
