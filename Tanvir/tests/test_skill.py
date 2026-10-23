"""Tests for the skill-level wriggle stack (skill_env, train_skill, eval_skill; WRIGGLE_FIT_RL.md §6.9).

    python -m pytest Tanvir/tests -q
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                       # noqa: E402

import socket_family as SF               # noqa: E402


def test_skill_env_shapes_and_finite():
    import skill_env as KE
    import socket_env as SE
    env = KE.WriggleSkillEnv(seed=1)
    obs = env.reset(context=SF.family(5, "SLOT"), level=2, brace="none")
    assert obs["actor"].shape == (KE.MACRO_DIM,)
    assert obs["critic"].shape == (KE.MACRO_DIM + SE.PRIV_DIM,)
    rng = np.random.default_rng(0)
    done, n = False, 0
    while not done and n < 6:
        obs, r, done, info = env.step(rng.uniform(-1, 1, KE.THETA_DIM))
        n += 1
        assert np.isfinite(obs["actor"]).all() and np.isfinite(obs["critic"]).all() and np.isfinite(r)
        assert obs["actor"].shape == (KE.MACRO_DIM,)
    assert n >= 1 and info["macro_steps"] == n
    # the contact happened before the first decision, and a macro step is K env ticks (or the episode's end)
    assert env.ctrl.contact and (done or env.env.steps >= KE.K_TICKS * n)


def test_theta_zero_is_the_scripted_controller():
    import skill_env as KE
    import socket_env as SE
    ctx = SF.family(2, "FREE", layer=0)
    for level in (0, 2):
        sc = SE.run_episode(SE.SocketInsertionEnv(mode="scripted", sensing="cheap", seed=10000),
                            context=ctx, level=level, brace="none")
        sk = KE.run_episode(KE.WriggleSkillEnv(seed=10000), None, context=ctx, level=level, brace="none")
        assert sk["reason"] == sc["reason"] and sk["steps"] == sc["steps"]
        assert abs(sk["peak_force_N"] - sc["peak_force_N"]) < 1e-6
    ctx = SF.family(2, "FREE", layer=0)
    sk = KE.run_episode(KE.WriggleSkillEnv(seed=10000), None, context=ctx, level=0, brace="none")
    assert sk["success"]                                   # an open socket at level 0 seats with theta = 0


def test_macro_reward_shaping_telescopes():
    import skill_env as KE
    kw = dict(context=SF.family(4, "SLOT"), level=2, brace="none")
    plain = KE.WriggleSkillEnv(seed=4, shaping=0.0)
    shaped = KE.WriggleSkillEnv(seed=4, shaping=0.5)
    plain.reset(**kw)
    shaped.reset(**kw)
    phi0 = shaped.phi
    assert 0.0 <= phi0 <= 0.5
    rng = np.random.default_rng(1)
    total, done, n = 0.0, False, 0
    while not done:                                        # same seed and thetas: identical trajectories
        th = rng.uniform(-1, 1, KE.THETA_DIM) * 0.5
        _, r0, done, _ = plain.step(th)
        _, r1, d1, _ = shaped.step(th)
        assert d1 == done
        total += (r1 - r0) * KE.GAMMA_MACRO ** n
        n += 1
    assert abs(total + phi0) < 1e-6                        # discounted shaping telescopes to -Phi(s0); Phi = 0 at the end
    assert shaped.phi == 0.0


def test_theta_channels_reach_the_controller():
    import skill_env as KE
    env = KE.WriggleSkillEnv(seed=3, shaping=0.0)
    env.reset(context=SF.family(5, "SLOT"), level=1, brace="none")
    tb = env.env.believed[0]
    c0 = env.ctrl.centre[:2].copy()
    th = np.zeros(KE.THETA_DIM)
    th[0], th[8] = 1.0, -1.0                               # nudge +x every macro step, soft laterally
    offs, done, n = [], False, 0
    while not done and n < 12:
        _, _, done, _ = env.step(th)
        n += 1
        offs.append(np.linalg.norm(env.ctrl.centre[:2] - tb[:2]))
        assert abs(env.env.k_scale - 0.15) < 1e-9 or env.env.released
    assert max(offs) <= KE.CENTRE_MAX + 1e-9               # the accumulated offset is clipped at 4 mm
    if n >= 2 and env.ctrl.phase == "search":
        assert env.ctrl.centre[0] > c0[0]                  # and it did accumulate along +x
    # the mapping: 0 -> scripted values at the centre of every range
    c = KE.SkillController(50.0, False)
    assert abs(c.rock_amp() - np.radians(1.5)) < 1e-12 and c.press_mult() == (1.0, 1.0)
    assert KE.pw(0.0, 0.15, 1.0, 1.0) == 1.0 and abs(KE.pw(-1.0, 0.15, 1.0, 1.0) - 0.15) < 1e-12


def test_checkpoint_zero_policy_is_theta_zero(tmp_path):
    import torch
    import skill_env as KE
    import train_skill as TS
    from tasks.ppo import ActorCritic, RunningNorm
    net = ActorCritic(KE.MACRO_DIM, KE.PRIV_DIM, KE.THETA_DIM, init_std=0.4)
    an, cn = RunningNorm(KE.MACRO_DIM), RunningNorm(KE.MACRO_DIM + KE.PRIV_DIM)
    TS.save(tmp_path / "c.pt", net, an, cn, {"skill": True}, 0, {})
    pol = TS.SkillPolicy(str(tmp_path / "c.pt"))
    assert np.allclose(pol(np.random.randn(KE.MACRO_DIM).astype(np.float32)), 0.0)


def test_es_zero_params_give_theta_zero():
    import es_skill as ES
    import skill_env as KE
    env = KE.WriggleSkillEnv(seed=2, shaping=0.0)
    obs = env.reset(context=SF.family(3, "POCKET"), level=2, brace="ideal")
    pol = ES.ESPolicy(np.zeros(ES.N_PARAMS))
    assert ES.N_PARAMS == 6 * 10 + 10 * 8
    for _ in range(3):
        th = pol(obs["actor"])
        assert np.array_equal(th, np.zeros(KE.THETA_DIM))
        obs, _, done, _ = env.step(th)
        if done:
            break
    f = ES.features(obs["actor"])
    assert f.shape == (8,) and np.isfinite(f).all()
    # a bias on one class moves only that class's theta
    p = np.zeros(ES.N_PARAMS)
    p[3 * 10 + 5] = 0.7                                   # class index 3 (SLOT), rock amplitude
    env.reset(context=SF.family(3, "SLOT"), level=0, brace="none")
    o = env._macro_obs(list(env.env.hist)[-KE.K_TICKS:])["actor"]
    assert abs(ES.ESPolicy(p)(o)[5] - 0.7) < 1e-12
    o2 = env.reset(context=SF.family(3, "FREE"), level=0, brace="none")["actor"]
    assert ES.ESPolicy(p)(o2)[5] == 0.0


def test_es_zero_episode_equals_scripted():
    import es_skill as ES
    import skill_env as KE
    ctx = SF.family(2, "FREE", layer=0)
    job = ("zero", np.zeros(ES.N_PARAMS), ctx, 0, "none", 10000, False)
    _, ret, ok, reason, cls = ES._job(job)
    ref = KE.run_episode(KE.WriggleSkillEnv(seed=10000, shaping=0.0), None, context=ctx, level=0, brace="none")
    assert ok == ref["success"] and reason == ref["reason"] and cls == "FREE"


def test_es_common_random_numbers():
    import es_skill as ES
    pool = SF.pool("train", 6)
    b1, b2 = ES.make_batch(pool, 7, seed=0), ES.make_batch(pool, 7, seed=0)
    assert [(c["id"], l, br, s) for c, l, br, s in b1] == [(c["id"], l, br, s) for c, l, br, s in b2]
    assert len(b1) == 6 * 4 and {c["class"] for c, *_ in b1} == set(SF.CLASSES)
    assert {(l) for _, l, _, _ in b1} == {0, 1, 2, 3} and {br for _, _, br, _ in b1} == {"none", "ideal"}
    b3 = ES.make_batch(pool, 8, seed=0)                    # a fresh batch each generation
    assert [s for *_, s in b1] != [s for *_, s in b3]
    cands = {"mean": np.zeros(ES.N_PARAMS), "+0": np.ones(ES.N_PARAMS), "-0": -np.ones(ES.N_PARAMS)}
    jobs = ES.build_jobs(cands, b1, True)
    per = {}
    for name, _, ctx, level, brace, seed, ps in jobs:
        per.setdefault(name, []).append((ctx["id"], level, brace, seed, ps))
    assert per["mean"] == per["+0"] == per["-0"] and len(per["mean"]) == len(b1)


def test_pad_shift_moves_the_pads_off_the_flush_neighbour():
    import socket_env as SE
    ctx = SF.family(100002, "FLUSH")                       # 2x2 target, neighbour on -y, pinch axis x
    off = SE.SocketInsertionEnv(mode="scripted", seed=10000)
    on = SE.SocketInsertionEnv(mode="scripted", seed=10000, pad_shift=True)
    off.reset(context=ctx, level=0, brace="none")
    on.reset(context=ctx, level=0, brace="none")
    assert on.axis == "x" and ctx["faces"] == ["-y"]
    ys_off = [off.m.geom_pos[g][1] for g in off.pads]
    ys_on = [on.m.geom_pos[g][1] for g in on.pads]
    assert ys_off == [0.0, 0.0]
    # the pad's -y edge now lies 0.2 mm inside the brick's -y face, i.e. it no longer overhangs the neighbour
    _, ly = SE.Bk.outer(2, 2)
    edge = ys_on[0] - SE.FINGER_W / 2
    assert abs(edge - (-ly / 2 + 0.0002)) < 1e-9 and ys_on[0] == ys_on[1] > 0
    # a free socket is untouched
    on.reset(context=SF.family(2, "FREE", layer=0), level=0, brace="none")
    assert [on.m.geom_pos[g][1] for g in on.pads] == [0.0, 0.0]


def test_pad_shift_seats_the_flush_sockets_that_the_overhang_blocked():
    import socket_env as SE
    ctx = SF.family(100002, "FLUSH")
    res = {}
    for ps in (False, True):
        env = SE.SocketInsertionEnv(mode="scripted", seed=10000, pad_shift=ps)
        res[ps] = SE.run_episode(env, None, context=ctx, level=0, brace="none")
    assert res[False]["reason"] == "joint_break" and res[True]["success"]
