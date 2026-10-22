"""Tests for the wriggle-fit stack (socket_family, wriggle_brace, socket_env, train_wriggle.Buckets).

    python -m pytest Tanvir/tests -q          # ~1 min on CPU
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                       # noqa: E402
import pytest                            # noqa: E402

import socket_family as SF               # noqa: E402
import wriggle_brace as WB               # noqa: E402
import planner as P                      # noqa: E402


@pytest.mark.parametrize("cls", SF.CLASSES)
def test_family_class_and_support(cls):
    for seed in range(15):
        ctx = SF.family(seed, cls)
        ts = SF.tuples(ctx)
        assert ctx["class"] == cls
        assert ts[-1][0] == ctx["target"] == "target"
        placed = []
        for t in ts:                                   # buildable in sequence order, no overlaps
            assert t[4] == 0 or P.supports(t, placed), (ctx["id"], t)
            for p in placed:
                assert not (P.cells(t)[0] & P.cells(p)[0] and t[4] == p[4]), (ctx["id"], t, p)
            placed.append(t)
        assert min(i for t in ts for i, _ in P.cells(t)[0]) == 0
        assert min(j for t in ts for _, j in P.cells(t)[0]) == 0


def test_family_deterministic_and_splits_disjoint():
    assert SF.family(5, "POCKET") == SF.family(5, "POCKET")
    train = {c["id"] for cs in SF.pool("train", 3).values() for c in cs}
    test = {c["id"] for cs in SF.pool("test", 3).values() for c in cs}
    assert not train & test


def test_repo_contexts_match_audit():
    rep = SF.repo_contexts()
    assert len(rep) == 34
    exec_ids = {(c["structure"], c["step"]) for c in rep if c["kind"] == "exec"}
    for s in [("S2", 6), ("S4", 3), ("S4", 7), ("S4", 10), ("S4", 13), ("S5", 4), ("hollow_box", 7), ("arch", 10)]:
        assert s in exec_ids
    pyr = sorted(c["step"] for c in rep if c["structure"] == "pyramid" and c["kind"] == "exec")
    assert pyr == [2, 3, 4, 6, 7, 8, 9]                      # plan_v4_rl_place fact 3
    assert {c["class"] for c in rep if c["kind"] == "last" and c["structure"] == "pyramid"} == {"POCKET"}
    arch = next(c for c in rep if c["id"] == "arch_b_010_exec")
    assert arch["planner_brace"] == "b_002"


def test_wriggle_brace_picks_the_neighbour_that_broke():
    u = WB.neighbour_utilisation(WB.probe_pocket())
    assert WB.choose(WB.probe_pocket())[0] == "nb2"          # the 1x2 that broke in every unbraced probe pocket
    assert u["nb0"][0] < WB.U_TRIGGER and u["nb1"][0] < WB.U_TRIGGER


def test_env_reset_step_and_release():
    import socket_env as SE
    env = SE.SocketInsertionEnv(mode="e2e", sensing="cheap", seed=0)
    obs = env.reset(context=SF.family(1, "ENCLOSED", layer=0), level=0, brace="none")
    assert obs["actor"].shape == (SE.OBS_DIM,) and obs["critic"].shape == (SE.OBS_DIM + SE.PRIV_DIM,)
    assert env.blocked                                    # enclosed: no free pinch axis
    x, _ = env.hand_pose()
    assert abs(x[2] - env.d.xpos[env.bid][2] - 0.012) < 2e-4   # brick hangs 12 mm under the TCP
    obs, r, done, info = env.step(np.zeros(SE.ACT_DIM))
    assert not done and np.isfinite(obs["actor"]).all()
    a = np.zeros(SE.ACT_DIM)
    a[8] = 1.0
    env.step(a)
    assert env.released and env.d.eq_active[env.eq] == 0
    assert all(env.m.geom_contype[g] == 0 for g in env.pads)


def test_cheap_sensing_is_quiet_in_free_space():
    import socket_env as SE
    env = SE.SocketInsertionEnv(mode="e2e", sensing="cheap", seed=3)
    env.reset(context=SF.family(2, "FREE", layer=0), level=0, brace="none")
    f = []
    hold = np.zeros(SE.ACT_DIM)
    hold[6] = -1.0                                        # no press: a 0 press channel is half the budget
    for _ in range(10):
        env.step(hold)
        f.append(np.linalg.norm(env.sensed[:3]))
        assert np.linalg.norm(env.ft_true[:3]) < 0.5      # truly in free space
    assert max(f) < 3.0                                   # below the scripted contact threshold
    assert np.std(f) > 0.0


def test_brace_stand_in_and_keepout():
    import socket_env as SE
    env = SE.SocketInsertionEnv(mode="e2e", sensing="cheap", seed=0)
    env.reset(context=WB.probe_pocket(), level=0, brace="ideal")
    assert env.braced == "nb2" and env.keepout is not None
    assert all(st.patch.f_break > 100 for st in env.cm.bricks["nb2"].conns)
    assert env.axis == "y"                                # no finger over the braced 1x2 on +x
    env.reset(context=WB.probe_pocket(), level=0, brace="none")
    assert env.braced is None and env.keepout is None


def test_buckets_unlock_frontier():
    from train_wriggle import Buckets, BUCKETS
    b = Buckets()
    assert b.weights().sum() == 1.0
    for _ in range(25):
        b.record({"class": "FREE", "level": 0, "success": True})
    b.update()
    assert (0, 1) in b.unlocked and (1, 0) in b.unlocked and (1, 1) not in b.unlocked
    assert len(BUCKETS) == 30


def test_buckets_unlock_gate():
    from train_wriggle import Buckets
    b = Buckets()
    for _ in range(25):
        b.record({"class": "FREE", "level": 0, "success": True})
    b.update(overall=0.3, gate=0.4)                       # overall too low: nothing opens
    assert b.unlocked == {(0, 0)}
    b.update(overall=0.5, gate=0.4)
    assert (0, 1) in b.unlocked


def test_residual_scale_and_release_lock():
    import socket_env as SE
    kw = dict(context=SF.family(2, "FREE", layer=0), level=0, brace="none")
    env = SE.SocketInsertionEnv(mode="residual", sensing="cheap", seed=0, res_scale=0.3, res_release=False)
    env.reset(**kw)
    a = np.zeros(SE.ACT_DIM)
    a[8] = 1.0                                            # the policy asks to release: ignored
    for _ in range(5):
        env.step(a)
    assert not env.released
    env2 = SE.SocketInsertionEnv(mode="residual", sensing="cheap", seed=0)
    env2.reset(**kw)
    env2.step(a)
    assert env2.released                                  # the old behaviour, still the default


def test_lost_brick_measures_the_brick_not_the_hand():
    import socket_env as SE
    env = SE.SocketInsertionEnv(mode="scripted", sensing="cheap", seed=0)
    env.reset(context=SF.family(1, "ENCLOSED", layer=0), level=3, brace="none")
    tx = env.true_tgt[0]
    assert abs(env._brick_err() - np.linalg.norm((env.d.xpos[env.bid] - tx)[:2])) < 1e-12
    assert abs(env._brick_err() - env._true_err()[0]) < 5e-4         # held: brick under the hand


def test_depth_shaping_is_potential_based():
    import socket_env as SE
    kw = dict(context=SF.family(2, "FREE", layer=0), level=0, brace="none")
    plain = SE.SocketInsertionEnv(mode="scripted", sensing="cheap", seed=4)
    shaped = SE.SocketInsertionEnv(mode="scripted", sensing="cheap", seed=4, shaping=0.5)
    plain.reset(**kw)
    shaped.reset(**kw)
    phi0 = shaped.phi
    assert 0.0 <= phi0 < 0.5
    total, done = 0.0, False
    while not done:                                       # same seed: identical trajectories
        _, r0, done, _ = plain.step(np.zeros(SE.ACT_DIM))
        _, r1, d1, _ = shaped.step(np.zeros(SE.ACT_DIM))
        assert d1 == done
        total += (r1 - r0) * SE.SHAPING_GAMMA ** (shaped.steps - 1)
    assert abs(total + phi0) < 1e-6                       # discounted shaping telescopes to -Phi(s0)
