import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import unittest

from Kean.insertion.env import FAIL, F_BREAK, F_INSERT, SUCCESS, InsertionEnv, press_n, residual
from Kean.insertion.obs import CAMERAS, from_shared, observe


def frames(n=8):
    return {cam: {"rgb": np.zeros((n, n, 3), np.uint8), "depth": np.ones((n, n), np.float32)} for cam in CAMERAS}


class Cell:
    def __init__(self):
        self.n_studs = 8
        self.force_n = 0.0
        self.snap = False
        self.broken = False
        self.dropped = False
        self.applied = None
        self.press = None

    def reset(self):
        self.applied = None

    def sense(self):
        return {
            "frames": frames(),
            "q": np.zeros(18),
            "qd": np.zeros(18),
            "hands": [np.zeros(7), np.zeros(7)],
            "stride": 1,
        }

    def scripted_action(self, step):
        action = np.zeros(7)
        action[2] = -0.2
        action[6] = 0.0
        return action

    def apply(self, command, press):
        self.applied = command
        self.press = press

    def privileged(self):
        return {"snap": self.snap, "broken": self.broken, "dropped": self.dropped, "force_n": self.force_n,
                "true_delta": np.zeros(3), "clutch": 0, "util": 0.0}


class InsertionTest(unittest.TestCase):
    def test_actor_obs_is_frames_and_proprio_only(self):
        obs = observe(frames(), np.zeros(18), np.zeros(18), [np.zeros(7), np.zeros(7)], stride=1)
        self.assertEqual(set(obs), {"frames", "proprio"})
        self.assertEqual(set(obs["frames"]), set(CAMERAS))
        self.assertEqual(obs["proprio"].shape, (54,))

    def test_shared_observation_filters_task_and_detection_state(self):
        images = {cam + "_" + key: value for cam, frame in frames().items()
                  for key, value in frame.items()}
        encoded = {"proprio": np.arange(54, dtype=np.float32), "images": images,
                   "bricks": "forbidden", "task": "forbidden", "privileged": "forbidden"}
        actor = from_shared(encoded)
        self.assertEqual(set(actor), {"proprio", "frames"})
        np.testing.assert_array_equal(actor["proprio"], encoded["proprio"])
        self.assertIs(actor["frames"]["top"]["rgb"], images["top_rgb"])
        # Do not rebuild the shared observation from state or call sensing again.
        cell = Cell()
        cell.observe = lambda: encoded
        cell.sense = lambda: (_ for _ in ()).throw(AssertionError("unexpected sensing"))
        self.assertEqual(set(InsertionEnv(cell).observe()), {"frames", "proprio"})

    def test_press_is_capped_at_insertion_budget(self):
        self.assertAlmostEqual(press_n(1.0, 8), 8 * F_INSERT)
        self.assertAlmostEqual(press_n(-1.0, 8), 0.0)
        self.assertLess(8 * F_INSERT, 8 * F_BREAK)

    def test_residual_cannot_exceed_unit_box(self):
        out = residual(np.ones(7), np.array([0.8, 0, 0, 0, 0, 0, 0.9]))
        self.assertTrue(np.all(np.abs(out) <= 1.0))
        self.assertAlmostEqual(out[6], 1.0)

    def test_sparse_reward(self):
        env = InsertionEnv(Cell())
        obs, reward, done, info = env.step(np.zeros(7))
        self.assertEqual(set(obs), {"frames", "proprio"})
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertAlmostEqual(info["press_n"], press_n(env.cell.applied[6], 8))
        env.cell.snap = True
        _, reward, done, _ = env.step(np.zeros(7))
        self.assertEqual((reward, done), (SUCCESS, True))
        env.cell.snap = False
        env.cell.force_n = 8 * F_BREAK + 1
        _, reward, done, _ = env.step(np.zeros(7))
        self.assertEqual((reward, done), (FAIL, True))

    def test_one_update_on_cpu(self):
        try:
            from Kean.insertion.train import train
        except ImportError:
            self.skipTest("torch is not installed")
        env = InsertionEnv(Cell())
        out = train(env, samples=4, horizon=2, device="cpu")
        self.assertEqual(out["samples"], 4)
        self.assertTrue(np.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()
