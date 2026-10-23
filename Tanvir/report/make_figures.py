"""Regenerate every figure of the Wriggle-Fit design proposal under Tanvir/report/figs/.

Run (single process, light):  OMP_NUM_THREADS=1 ~/.venv/bin/python Tanvir/report/make_figures.py
Data come only from files in the repo (Tanvir/results/*, Tanvir/socket_family.py contexts).
Diagrams (flow, primitive, timeline, reward) are schematic and say so in the report captions.
"""
import json
import math
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TANVIR = HERE.parent
RES = TANVIR / "results"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(TANVIR))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle, FancyArrowPatch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
C = dict(grey="#6b7280", red="#c0392b", blue="#2563eb", green="#16a34a", purple="#9333ea",
         orange="#ea7a0c", light="#e5e7eb", dark="#1f2937")


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. socket contexts (3D)
HOVER = 22.0                                              # mm: the target is drawn lifted above its seat
FACES_IDX = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]


def box(store, x0, y0, z0, dx, dy, dz, color, edge="#374151", lw=0.4):
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    for f in FACES_IDX:
        store.append(([v[i] for i in f], color, edge, lw))


def draw_context(ax, ctx, title):
    import planner as P
    pitch, h = 8.0, 9.6                                  # mm (masterplan 2.3)
    allc, store, plate, outl = [], [], [], []
    for b in ctx["bricks"]:
        t = (b["id"], b["type"], *b["grid_pos"], b["yaw_index"])
        allc += sorted(P.cells(t)[0])
    imax = max(i for i, _ in allc) + 1
    jmax = max(j for _, j in allc) + 1
    layer_cols = ["#93c5fd", "#a7d7a0", "#fcd34d"]
    ktop = 0
    for b in ctx["bricks"]:
        t = (b["id"], b["type"], *b["grid_pos"], b["yaw_index"])
        cells = sorted(P.cells(t)[0])
        i0, j0 = min(i for i, _ in cells), min(j for _, j in cells)
        ni = max(i for i, _ in cells) - i0 + 1
        nj = max(j for _, j in cells) - j0 + 1
        k = b["grid_pos"][2]
        ktop = max(ktop, k + 1)
        tgt = b["id"] == ctx["target"]
        col = "#f97316" if tgt else layer_cols[k % 3]
        x0, y0, z0 = i0 * pitch + 0.1, j0 * pitch + 0.1, k * h + (HOVER if tgt else 0.0)
        if tgt:
            outl.append((x0, y0, k * h, ni * pitch - 0.2, nj * pitch - 0.2))
        box(store, x0, y0, z0, ni * pitch - 0.2, nj * pitch - 0.2, h, col,
            edge="#7c2d12" if tgt else "#374151", lw=1.0 if tgt else 0.4)
        for (i, j) in cells:                              # studs
            box(store, i * pitch + 2.6, j * pitch + 2.6, z0 + h, 2.8, 2.8, 1.8, col, edge="#374151", lw=0.2)
    pad = 3.0
    box(plate, -pad, -pad, -3, imax * pitch + 2 * pad, jmax * pitch + 2 * pad, 3, "#e5e7eb", edge="#9ca3af", lw=0.3)
    # painter's order: far faces first (view azim -58, elev 38)
    az, el = math.radians(-58), math.radians(38)
    d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    store.sort(key=lambda f: float(np.mean(np.array(f[0]), axis=0) @ d))
    pc = Poly3DCollection([f[0] for f in store], facecolors=[f[1] for f in store],
                          edgecolors=[f[2] for f in store], linewidths=[f[3] for f in store])
    pp = Poly3DCollection([f[0] for f in plate], facecolors=[f[1] for f in plate],
                          edgecolors=[f[2] for f in plate], linewidths=0.3)
    ax.computed_zorder = False
    pp.set_zorder(1); pc.set_zorder(3)
    ax.add_collection3d(pp); ax.add_collection3d(pc)
    for (x0, y0, z0, dx, dy) in outl:                      # the seat, dashed
        ax.plot([x0, x0 + dx, x0 + dx, x0, x0], [y0, y0, y0 + dy, y0 + dy, y0], [z0] * 5,
                color="#7c2d12", ls="--", lw=0.9, zorder=2)
    W, D, H = imax * pitch + 2 * pad, jmax * pitch + 2 * pad, ktop * h + 5 + HOVER
    ax.set_xlim(-pad, -pad + W); ax.set_ylim(-pad, -pad + D); ax.set_zlim(-3, H)
    ax.set_box_aspect((W, D, H + 3))
    ax.view_init(elev=38, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8.5)


def fig_contexts():
    import socket_family as SF
    reps = {c["id"]: c for c in SF.repo_contexts()}
    panels = [
        (SF.family(200001, "FLUSH", 0), "FLUSH (1 wall)\nfamily test seed 200001"),
        (reps["S4_b_002_exec"], "SLOT (2 opposite walls)\nrepo: S4 step 3"),
        (SF.family(200002, "SLOT", 0), "SLOT, 1x2 neighbours\nfamily test seed 200002"),
        (SF.family(200000, "POCKET", 0), "POCKET (3 walls)\nfamily test seed 200000"),
        (SF.family(200000, "ENCLOSED", 0), "ENCLOSED (4 walls)\nfamily test seed 200000"),
        (reps["pyramid_b_003_exec"], "Finger-blocked step\nrepo: pyramid step 3 (CORNER)"),
    ]
    fig = plt.figure(figsize=(7.2, 5.0))
    for n, (ctx, t) in enumerate(panels):
        ax = fig.add_subplot(2, 3, n + 1, projection="3d")
        draw_context(ax, ctx, t)
    fig.text(0.5, 0.01, "orange = target brick, drawn 22 mm above its seat (dashed outline); blue/green/yellow = placed bricks, layers 0/1/2",
             ha="center", fontsize=8)
    fig.subplots_adjust(wspace=0.0, hspace=0.12)
    save(fig, "fig01_socket_contexts.png")


def fig_s5_context():
    """A load-bearing executed slot with supports (S5 step 4) next to the open version."""
    import socket_family as SF
    reps = {c["id"]: c for c in SF.repo_contexts()}
    fig = plt.figure(figsize=(3.6, 2.8))
    ax = fig.add_subplot(111, projection="3d")
    draw_context(ax, reps["S5_b_003_exec"], "SLOT on layer 1, repo: S5 step 4\n(21 bricks are built on it later)")
    save(fig, "fig01b_s5_slot.png")


# ---------------------------------------------------------------- 2. slot geometry to scale
def fig_gap():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    a = ax[0]
    a.add_patch(Rectangle((-12, -4), 4.0, 8, color=C["grey"]))
    a.add_patch(Rectangle((8.0, -4), 4.0, 8, color=C["grey"]))
    a.add_patch(Rectangle((-7.9, -3), 15.8, 6, color=C["orange"], alpha=0.85))
    a.text(0, 0, "brick 15.8 mm\n(2 studs: 8n - 0.2)", ha="center", va="center", fontsize=8)
    a.text(-10, 0, "neighbour", rotation=90, ha="center", va="center", color="white", fontsize=8)
    a.text(10, 0, "neighbour", rotation=90, ha="center", va="center", color="white", fontsize=8)
    a.annotate("", xy=(-8.0, -3.8), xytext=(-7.9, -3.8), arrowprops=dict(arrowstyle="<->"))
    a.text(-8.0, -3.95, "0.1 mm play\nper side", fontsize=7, ha="left", va="top")
    a.set_xlim(-12, 12); a.set_ylim(-6, 4.5); a.set_aspect("equal")
    a.set_title("Top view of a SLOT, to scale (mm)", fontsize=9)
    a.set_xlabel("x (mm)"); a.set_yticks([])
    b = ax[1]
    b.add_patch(Rectangle((-0.1, -1.5), 0.2, 3.0, color=C["orange"], alpha=0.25))
    b.add_patch(Rectangle((-0.1, -1.5), 0.2, 3.0, fill=False, ec=C["orange"]))
    for r, lab, ls in [(0.8, "aim error 1 sigma 0.8 mm", "-"), (2.0, "curriculum level 2 mm", "--"),
                       (4.0, "level 4 mm", ":")]:
        b.add_patch(Circle((0, 0), r, fill=False, ec=C["blue"], ls=ls, lw=1.2, label=lab))
    b.add_patch(Rectangle((-0.1, -1.5), 0.2, 3.0, color=C["orange"], alpha=0.25, label="play window, 0.2 mm wide (x)"))
    b.legend(fontsize=6.5, frameon=False, loc="upper right", bbox_to_anchor=(1.12, 1.0))
    b.set_xlim(-4.6, 4.6); b.set_ylim(-4.6, 4.6); b.set_aspect("equal")
    b.set_title("Aim error against the play (to scale)", fontsize=9)
    b.set_xlabel("x (mm)"); b.set_ylabel("y (mm)")
    fig.tight_layout()
    save(fig, "fig02_gap_vs_aim.png")


# ---------------------------------------------------------------- 3. dual-arm flow
def rbox(ax, x, y, w, h, text, fc="#f3f4f6", ec=C["dark"], fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, text=None, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=C["dark"], lw=1, **kw))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, fontsize=7, ha="center")


def fig_flow():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
    rbox(ax, 0.1, 3.2, 1.9, 0.9, "next plan step\n(brick, socket)", fs=7)
    rbox(ax, 2.6, 3.2, 2.3, 0.9, "tight_fit_audit:\nsocket class,\nfinger access", fc="#e0f2fe", fs=7)
    arrow(ax, (2.0, 3.65), (2.6, 3.65))
    rbox(ax, 5.5, 5.3, 3.4, 0.9, "REGULAR: FREE, FLUSH,\nCORNER, fingers free", fc="#dcfce7", fs=7)
    rbox(ax, 9.6, 5.3, 4.2, 0.9, "normal insert\n(existing scripted / RL inserter)", fc="#dcfce7", fs=7)
    arrow(ax, (4.9, 4.0), (5.5, 5.5)); arrow(ax, (8.9, 5.75), (9.6, 5.75))
    rbox(ax, 5.5, 3.2, 3.4, 0.9, "CRITICAL: SLOT, POCKET,\nENCLOSED, finger-blocked", fc="#fee2e2", fs=7)
    arrow(ax, (4.9, 3.65), (5.5, 3.65))
    ax.text(0.1, 2.3, "ARM A (stabiliser)", fontsize=8, weight="bold", color=C["blue"])
    ax.text(0.1, 0.9, "ARM B (placer)", fontsize=8, weight="bold", color=C["orange"])
    rbox(ax, 2.7, 1.8, 4.8, 0.95, "1. brace: planner's weakest-joint brace,\nor wriggle brace on the most fragile\nneighbour (utilisation >= 0.6)", fc="#dbeafe", fs=6.5)
    rbox(ax, 7.9, 1.8, 2.5, 0.95, "2. settle; then a\nstatic obstacle\nfor arm B", fc="#dbeafe", fs=6.5)
    rbox(ax, 11.0, 1.8, 2.8, 0.95, "4. unbrace after\nrelease and retreat", fc="#dbeafe", fs=6.5)
    arrow(ax, (7.2, 3.2), (5.2, 2.75)); arrow(ax, (7.5, 2.27), (7.9, 2.27)); arrow(ax, (10.4, 2.27), (11.0, 2.27))
    rbox(ax, 2.7, 0.2, 2.9, 0.95, "scripted approach\nto first contact", fc="#ffedd5", fs=6.5)
    rbox(ax, 6.0, 0.2, 4.3, 0.95, "3. WRIGGLE-FIT skill: tilt-in, square\nyaw, rock, press, soft lateral", fc="#fed7aa", fs=6.5)
    rbox(ax, 10.7, 0.2, 3.1, 0.95, "seated? release, press\nstuds (else abort, log)", fc="#ffedd5", fs=6.5)
    arrow(ax, (9.1, 1.8), (4.3, 1.15)); arrow(ax, (5.6, 0.67), (6.0, 0.67)); arrow(ax, (10.3, 0.67), (10.7, 0.67))
    arrow(ax, (12.4, 1.15), (12.4, 1.8))
    ax.text(7.0, 6.45, "Dual-arm flow at one build step", fontsize=9, weight="bold", ha="center")
    save(fig, "fig03_dualarm_flow.png")


# ---------------------------------------------------------------- 4. wriggle primitive
def brick_poly(cx, cy, w, h, deg):
    a = math.radians(deg)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return pts @ R.T + [cx, cy]


def fig_primitive():
    gap = 10.0
    stages = [
        ("1. approach", dict(c=(3.0, 8.2), deg=0, arrow=None), "brick above the slot,\noff by the aim error"),
        ("2. tilt-in", dict(c=(1.0, 4.8), deg=-14, arrow=("rot", -1)), "lead with one edge\ninto the slot mouth"),
        ("3. catch and slide", dict(c=(-0.2, 1.27), deg=-14, arrow=("x", -1)), "edge rests on the wall top;\nslide until it drops in"),
        ("4. drop and level", dict(c=(0.0, -1.2), deg=-3, arrow=("rot", 1)), "edge falls in; rotate level,\nlateral stiffness low"),
        ("5. press", dict(c=(0.0, -6.0), deg=0, arrow=("z", -1)), "walls centre the brick;\npress to the budget"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.5), sharey=True)
    for ax, (name, d, note) in zip(axes, stages):
        ax.add_patch(Rectangle((-gap / 2 - 6, -12), 6, 12, color=C["grey"]))
        ax.add_patch(Rectangle((gap / 2, -12), 6, 12, color=C["grey"]))
        ax.add_patch(Rectangle((-gap / 2 - 6, -12), 12 + gap, 0.0, color="none"))
        ax.add_patch(Polygon(brick_poly(d["c"][0], d["c"][1], 9.6, 5.0, d["deg"]), fc=C["orange"], ec="#7c2d12", alpha=0.9))
        if d["arrow"]:
            k, s = d["arrow"]
            cx, cy = d["c"]
            if k == "x":
                ax.annotate("", xy=(cx + s * 4, cy + 4.2), xytext=(cx - s * 0.5, cy + 4.2), arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.5))
            elif k == "z":
                ax.annotate("", xy=(cx, cy + 2.8), xytext=(cx, cy + 7.5), arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.5))
            else:
                ax.annotate("", xy=(cx + 5.5, cy + 4.2), xytext=(cx - 1.0, cy + 5.8), arrowprops=dict(arrowstyle="->", color=C["blue"], lw=1.5, connectionstyle="arc3,rad=-0.4"))
        ax.set_title(name, fontsize=8)
        ax.text(0, -15.8, note, ha="center", va="top", fontsize=6.5)
        ax.set_xlim(-12, 12); ax.set_ylim(-13, 13); ax.set_aspect("equal"); ax.axis("off")
    fig.text(0.5, 0.0, "Schematic side view, not to scale (the real play is 0.1 mm per side). Blue arrows: commanded motion.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig04_wriggle_primitive.png")


# ---------------------------------------------------------------- 5. timeline: per-step vs skill-level
def fig_timeline():
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True)
    T = 10.0
    a = ax[0]
    for t in np.arange(0, T, 0.05):
        a.plot([t, t], [0, 1], color=C["grey"], lw=0.4)
    a.axvline(T, color=C["red"], lw=2)
    a.text(T - 0.1, 1.25, "sparse reward\n(+1 / -1, once)", color=C["red"], ha="right", fontsize=8)
    a.set_ylim(0, 2); a.set_yticks([])
    a.set_title("Old: residual PPO on a 20 Hz scripted wriggle. One action per 50 ms, up to 200 per episode", fontsize=8.5, loc="left")
    a.spines["left"].set_visible(False)
    b = ax[1]
    b.add_patch(Rectangle((0, 0.2), 1.5, 0.8, color=C["light"], ec=C["dark"]))
    b.text(0.75, 0.6, "scripted\napproach", ha="center", va="center", fontsize=7)
    for k, t in enumerate(np.arange(1.5, T, 0.5)):
        b.add_patch(Rectangle((t, 0.2), 0.5, 0.8, color=C["orange"] if k % 2 == 0 else "#fdba74", ec=C["dark"], lw=0.5))
        b.plot([t], [1.1], marker="v", color=C["blue"], ms=4)
    b.text(5.7, 1.5, "policy picks the primitive parameters here (blue marks), every 0.5 s", fontsize=8, color=C["blue"], ha="center")
    b.axvline(T, color=C["red"], lw=2)
    b.set_ylim(0, 2); b.set_yticks([])
    b.set_title("New: skill-level (option) RL. About 15-20 decisions per episode; zero action = scripted-like wriggle", fontsize=8.5, loc="left")
    b.set_xlabel("time in episode (s); the episode times out at 10 s")
    b.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig05_timeline.png")


# ---------------------------------------------------------------- 6. reward / termination
def fig_reward():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    rbox(ax, 0.1, 3.4, 3.6, 1.4, "scripted approach, then\na decision every 0.5 s\n(10 ticks of 50 ms)", fc="#f3f4f6", fs=7.5)
    rbox(ax, 0.1, 1.2, 5.6, 1.6, "per tick:  -0.002 (finish quickly)\n-0.05 * max(0, u_nb - 0.6)\nu_nb = highest neighbour-joint utilisation\n(privileged, used in the reward only)", fc="#e0f2fe", fs=7.5)
    arrow(ax, (1.9, 3.4), (1.9, 2.8))
    ax.add_patch(FancyBboxPatch((6.6, 0.3), 7.3, 5.4, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=C["grey"], ls="--"))
    ax.text(10.25, 5.35, "terminal events end the episode", fontsize=8, style="italic", ha="center")
    terms = [("+1  seated (clutch reaches MATED)", "#dcfce7"), ("-1  neighbour joint breaks or moves > 0.3 mm", "#fee2e2"),
             ("-1  force over budget", "#fee2e2"), ("-1  hand or brick touches arm A", "#fee2e2"),
             ("-1  released brick > 3 mm from its seat", "#fee2e2"), ("timeout at 10 s: no terminal bonus", "#f3f4f6")]
    for n, (t, c) in enumerate(terms):
        rbox(ax, 6.9, 4.45 - n * 0.78, 6.7, 0.6, t, fc=c, fs=7.5)
    arrow(ax, (3.7, 4.1), (6.6, 3.0))
    ax.text(0.1, 0.75, "Ablation only: potential shaping gamma*Phi(s') - Phi(s),", fontsize=7)
    ax.text(0.1, 0.4, "Phi = k * depth progress, 0 at episode end", fontsize=7)
    ax.text(0.1, 0.05, "(the optimal policy is unchanged).", fontsize=7)
    save(fig, "fig06_reward.png")


# ---------------------------------------------------------------- 7. slot probe chart
def parse_md_rows(path, header_startswith):
    rows, on = [], False
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("|") and header_startswith in ln:
            on = True
            continue
        if on:
            if not ln.startswith("|"):
                break
            if ln.startswith("|---"):
                continue
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows


def fig_slot_probe():
    rows = parse_md_rows(RES / "slot_probe.md", "| socket |")
    data = {}
    for r in rows:
        s, br, e, seated = r[0], r[1], float(r[2]), r[3]
        n, d = seated.split("/")
        brk = 0
        m = re.search(r"joint_break (\d+)", r[8])
        if m:
            brk = int(m.group(1))
        data[(s, br, e)] = (int(n), int(d), brk)
    errs = [0, 1, 2, 3, 4]
    series = [("open, no brace", ("open", "none"), C["green"], "o", "-"),
              ("pocket, no brace", ("pocket", "none"), C["red"], "s", "-"),
              ("pocket, braced (10x joints)", ("pocket", "all"), C["blue"], "^", "--")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for lab, (s, br), col, mk, ls in series:
        ax[0].plot(errs, [data[(s, br, float(e))][0] / data[(s, br, float(e))][1] * 100 for e in errs],
                   color=col, marker=mk, ls=ls, label=lab)
    ax[0].set_xlabel("aim error (mm)"); ax[0].set_ylabel("seated (%) of 6")
    ax[0].set_title("Scripted base, 2x4 brick", fontsize=9); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_ylim(-3, 105)
    w = 0.38
    ax[1].bar(np.array(errs) - w / 2, [data[("pocket", "none", float(e))][2] for e in errs], w, color=C["red"], label="no brace")
    ax[1].bar(np.array(errs) + w / 2, [data[("pocket", "all", float(e))][2] for e in errs], w, color=C["blue"], label="braced")
    ax[1].set_xlabel("aim error (mm)"); ax[1].set_ylabel("joint breaks (of 6)")
    ax[1].set_title("Pocket: neighbour joint breaks", fontsize=9); ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig07_slot_probe.png")


# ---------------------------------------------------------------- 8. audit chart
def fig_audit():
    rows = parse_md_rows(RES / "tight_fit_audit.md", "| structure |")
    names, cls, last = [], [], []
    for r in rows:
        if r[0].startswith("corbel"):
            continue
        names.append(r[0]); cls.append([int(x) for x in r[3:9]])
        m = re.match(r"(\d+)", r[11])
        last.append(int(m.group(1)) if m else 0)
    cls = np.array(cls)
    labels = ["FREE", "FLUSH", "CORNER", "SLOT", "POCKET", "ENCLOSED"]
    cols = ["#d1d5db", "#93c5fd", "#a7d7a0", "#f97316", "#dc2626", "#7f1d1d"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(names)); left = np.zeros(len(names))
    for k in range(6):
        ax.barh(y, cls[:, k], left=left, color=cols[k], label=labels[k], ec="white", lw=0.4)
        left += cls[:, k]
    for i, l in enumerate(last):
        ax.text(left[i] + 0.4, i, "+%d if placed last in layer" % l if l else "", va="center", fontsize=6.5, color=C["dark"])
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("bricks (class as built, in the simulator's step order)")
    ax.legend(ncol=6, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    ax.set_xlim(0, 33)
    fig.tight_layout()
    save(fig, "fig08_audit.png")


# ---------------------------------------------------------------- 9. scripted baseline by aim error
def fig_baseline():
    rows = parse_md_rows(RES / "eval" / "scripted_cheap_v2.md", "| class |")
    classes, v = [], []
    for r in rows:
        classes.append(r[0])
        v.append([int(x.split("/")[0]) / int(x.split("/")[1]) * 100 for x in r[1:4]])
    v = np.array(v)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    x = np.arange(len(classes)); w = 0.26
    for k, (lab, col) in enumerate([("aim <= 0.5 mm", C["green"]), ("aim 1-2 mm", C["orange"]), ("aim 3-4 mm", C["red"])]):
        ax.bar(x + (k - 1) * w, v[:, k], w, color=col, label=lab)
        for xi, val in zip(x, v[:, k]):
            ax.text(xi + (k - 1) * w, val + 1, "%d" % round(val), ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("seated (%)"); ax.set_ylim(0, 75); ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Scripted wriggle, cheap sensing, all brace conditions pooled (scripted_cheap_v2.md)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig09_scripted_baseline.png")


# ---------------------------------------------------------------- 10. training curves
def curve(name, k=1):
    xs, ys = [], []
    for ln in (RES / "runs" / name / "train.jsonl").read_text().splitlines():
        d = json.loads(ln)
        xs.append(d["samples"]); ys.append(d["success_200"])
    return np.array(xs), np.array(ys)


def fig_curves():
    runs = [("main (unsafeguarded residual, 1.5 M)", "main", C["red"]),
            ("safe_short (L2 0.01)", "safe_short", C["purple"]),
            ("safe_short2 (L2 0.1)", "safe_short2", C["blue"]),
            ("safe_shaped (L2 0.1 + depth shaping)", "safe_shaped", C["green"])]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for lab, n, col in runs:
        x, y = curve(n)
        ax[0].plot(x / 1e3, y, color=col, label=lab, lw=1.3)
        if n != "main":
            ax[1].plot(x / 1e3, y, color=col, lw=1.3)
        else:
            m = x <= 300000
            ax[1].plot(x[m] / 1e3, y[m], color=col, lw=1.3)
    ax[0].set_xlabel("training samples (thousands)"); ax[0].set_ylabel("success, last 200 training episodes")
    ax[0].set_ylim(0, 1); ax[0].set_title("All runs", fontsize=9)
    ax[1].set_xlabel("training samples (thousands)"); ax[1].set_ylim(0, 1); ax[1].set_title("First 300 k samples", fontsize=9)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig10_training_curves.png")


# ---------------------------------------------------------------- 11. validation selection
def fig_selection():
    names = [("safe_short", C["purple"], "L2 0.01"), ("safe_short2", C["blue"], "L2 0.1"),
             ("safe_shaped", C["green"], "L2 0.1 + shaping")]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ck = ["0 (scripted)", "100 k", "200 k", "300 k"]
    w = 0.26
    for k, (n, col, lab) in enumerate(names):
        sel = json.loads((RES / "runs" / n / "select.json").read_text())
        vals = [r["seated"] for r in sel["results"]]
        eps = sel["results"][0]["episodes"]
        ax.bar(np.arange(4) + (k - 1) * w, vals, w, color=col, label="%s (%s)" % (n, lab))
        for i, val in enumerate(vals):
            ax.text(i + (k - 1) * w, val + 0.5, str(val), ha="center", fontsize=7)
    ax.axhline(37, color=C["dark"], ls=":", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(ck)
    ax.set_xlabel("checkpoint (samples)"); ax.set_ylabel("seated of %d validation episodes" % eps)
    ax.set_ylim(0, 46); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center")
    ax.set_title("Validation selection: no trained checkpoint beats the scripted base (select.json)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig11_selection.png")


# ---------------------------------------------------------------- 12. curriculum reach
def fig_buckets():
    classes = ["FREE", "FLUSH", "CORNER", "SLOT", "POCKET", "ENCLOSED"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for a, (n, t) in zip(ax, [("main", "main (1.5 M)"), ("safe_shaped", "safe_shaped (300 k)")]):
        b = json.loads((RES / "runs" / n / "buckets.json").read_text())
        M = np.full((6, 5), np.nan)
        for i, c in enumerate(classes):
            for L in range(5):
                v = b["%s/%d" % (c, L)]
                if v["unlocked"] and v["rate"] is not None:
                    M[i, L] = v["rate"] * 100
        im = a.imshow(M, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
        for i in range(6):
            for L in range(5):
                a.text(L, i, "-" if np.isnan(M[i, L]) else "%d" % M[i, L], ha="center", va="center", fontsize=7)
        a.set_xticks(range(5)); a.set_xticklabels(["0.5", "1", "2", "3", "4"])
        a.set_yticks(range(6)); a.set_yticklabels(classes)
        a.set_xlabel("aim-error level (mm)"); a.set_title(t, fontsize=9)
        for s in a.spines.values():
            s.set_visible(False)
    fig.text(0.5, -0.04, "Final training success (%) per bucket; '-' = bucket never unlocked (buckets.json)", ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig12_buckets.png")


# ---------------------------------------------------------------- 13. session results
def _seated(path, src, brace, level):
    n = d = 0
    for ln in Path(path).read_text().splitlines():
        r = json.loads(ln)
        cid = str(r["context"]).split("_")[0]
        is_repo = cid in ("S2", "S4", "S5", "S2A", "hollow", "arch", "pyramid")
        s_ = "repo" if is_repo else ("fam" if cid == "fam" else "probe")
        if s_ == src and r["brace"] == brace and r["level"] == level:
            d += 1; n += bool(r["success"])
    return n, d


def fig_session():
    old = RES / "eval" / "scripted_cheap_v2.jsonl"
    new = RES / "eval" / "es_main_zero.jsonl"
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    lv = [0, 2, 4]; x = np.arange(3); w = 0.36
    for k, (pth, lab, col) in enumerate([(old, "centred pads", C["grey"]), (new, "pad shift", C["green"])]):
        v = [_seated(pth, "repo", "none", L) for L in lv]
        pc = [100 * n / d for n, d in v]
        ax[0].bar(x + (k - 0.5) * w, pc, w, color=col, label=lab)
        for xi, (n, d), p_ in zip(x, v, pc):
            ax[0].text(xi + (k - 0.5) * w, p_ + 1, "%d/%d" % (n, d), ha="center", fontsize=6)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["<= 0.5", "1-2", "3-4"]); ax[0].set_xlabel("aim error (mm)")
    ax[0].set_ylabel("seated (%)"); ax[0].set_ylim(0, 70); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("Held-out repo, no brace", fontsize=8.5)
    gens, diff = [], []
    for ln in (RES / "runs" / "es_main" / "train.jsonl").read_text().splitlines():
        d = json.loads(ln)
        if d.get("paired_diff") is not None:
            gens.append(d["gen"]); diff.append(d["paired_diff"])
    ax[1].axhline(0, color=C["dark"], lw=0.8)
    ax[1].plot(gens, diff, color=C["blue"], marker="o")
    ax[1].set_xlabel("ES generation"); ax[1].set_ylabel("fitness: mean minus theta 0")
    ax[1].set_title("ES, paired on the same 24 episodes", fontsize=8.5)
    sel = json.loads((RES / "runs" / "es_main" / "select.json").read_text())["results"]
    names = ["th=0", "g11", "g12", "g13", "g14", "avg"]
    vals = [r["seated"] for r in sel]
    ax[2].bar(range(len(vals)), vals, color=[C["green"]] + [C["blue"]] * (len(vals) - 1))
    for i, v_ in enumerate(vals):
        ax[2].text(i, v_ + 1, str(v_), ha="center", fontsize=7)
    ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels(names, fontsize=7)
    ax[2].set_ylabel("seated of 120"); ax[2].set_ylim(0, 80); ax[2].set_title("ES validation", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig13_session_results.png")


if __name__ == "__main__":
    fig_contexts(); fig_s5_context(); fig_gap(); fig_flow(); fig_primitive(); fig_timeline(); fig_reward()
    fig_slot_probe(); fig_audit(); fig_baseline(); fig_curves(); fig_selection(); fig_buckets(); fig_session()
