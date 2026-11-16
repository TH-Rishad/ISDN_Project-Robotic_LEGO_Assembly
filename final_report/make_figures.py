"""Data figures for the final report, computed from the code and saved run files.

    Sakib/.venv/bin/python final_report/make_figures.py

Simulation stills come from render_stills.py (needs the GPU); this script is CPU only.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "Sakib" / "brickassembly"
sys.path.insert(0, str(CELL))
import blueprint as B  # noqa: E402
import planner as P  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
RUN = ROOT / "Kean" / "results" / "dual_arch_s3_012"
RUN_PYR = ROOT / "Kean" / "results" / "dual_pyramid_s3_006"
USER = B.SAMPLE_DIR / "user"
PAL = ["#d62728", "#1f77b4", "#f2b701", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
       "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---- 1. inputs -------------------------------------------------------------
for src in (B.SAMPLE_DIR / "arch_front.png", USER / "pyramid_front.png", USER / "pyramid_side.png",
            B.SAMPLE_DIR / "cube_front.png", B.SAMPLE_DIR / "cube_side.png"):
    shutil.copy(src, OUT / ("input_" + src.name))


# ---- 2. drawing -> grid -> voxels -> bricks -----------------------------------
def brick_colors(bricks):
    return {b[0]: PAL[n % len(PAL)] for n, b in enumerate(bricks)}


def draw_bricks3d(ax, bricks, numbers=False):
    """Bricks as voxels, one colour per brick, labelled with build order."""
    NI = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    NJ = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    NK = max(b[4] for b in bricks) + 1
    filled = np.zeros((NJ, NI, NK), bool)       # plot j along x so the front view faces the reader
    colors = np.empty(filled.shape, object)
    col = brick_colors(bricks)
    for n, b in enumerate(bricks):
        cells, k = P.cells(b)
        for i, j in cells:
            filled[j, i, k] = True
            colors[j, i, k] = col[b[0]]
        if numbers:
            ci = np.mean([c[0] for c in cells]) + .5
            cj = np.mean([c[1] for c in cells]) + .5
            ax.text(cj, -0.15, k + .5, str(n), fontsize=7, ha="center", va="center", zorder=10,
                    color="k", fontweight="bold")
    ax.voxels(filled, facecolors=colors, edgecolor="k", linewidth=.4)
    ax.set_box_aspect((NJ, NI, NK * P.BRICK_H / P.PITCH))
    ax.view_init(elev=22, azim=-62)


def pipeline_figure(name, front, side=None, width=None, depth=None):
    f = B.mask(front)
    V = B.carve(front, side, None, width, depth)
    NI, NJ, NK = V.shape
    bricks = B.tile(V)
    F = B.resample(f, NK, NJ)
    fig = plt.figure(figsize=(13, 3.4))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(f, cmap="gray_r", interpolation="nearest")
    ax.set_title("1. Silhouette (cropped)")
    ax.axis("off")
    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(F, cmap="Blues", vmin=0, vmax=1.4, extent=(0, NJ, 0, NK))
    for a in range(NJ + 1):
        ax.axvline(a, color="k", lw=.5)
    for a in range(NK + 1):
        ax.axhline(a, color="k", lw=.5)
    ax.set_title("2. Stud grid, majority vote")
    ax.set_xlabel("studs (y)")
    ax.set_ylabel("layer")
    ax.set_aspect(P.PITCH / P.BRICK_H)
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    vox = np.transpose(V, (1, 0, 2))
    ax.voxels(vox, facecolors="#9ecae1", edgecolor="#3182bd", linewidth=.3)
    ax.set_box_aspect((NJ, NI, NK * P.BRICK_H / P.PITCH))
    ax.view_init(elev=22, azim=-62)
    ax.set_title("3. Visual hull (%d cells)" % V.sum())
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    draw_bricks3d(ax, bricks)
    ax.set_title("4. Tiled into %d bricks" % len(bricks))
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    save(fig, name)
    return bricks


arch = pipeline_figure("pipeline_arch.png", B.SAMPLE_DIR / "arch_front.png", depth=2)
pyramid = pipeline_figure("pipeline_pyramid.png", USER / "pyramid_front.png", USER / "pyramid_side.png", width=5)


def layers_figure(name, bricks, title):
    """Top-down view of each layer: which brick covers which stud."""
    NI = max(b[2] + P.footprint(b[1], b[5])[0] for b in bricks)
    NJ = max(b[3] + P.footprint(b[1], b[5])[1] for b in bricks)
    NK = max(b[4] for b in bricks) + 1
    col = brick_colors(bricks)
    fig, axs = plt.subplots(1, NK, figsize=(2.3 * NK, 1.0 + 1.9 * NI / max(NJ, 1) * 1.2))
    for k, ax in enumerate(np.atleast_1d(axs)):
        for n, b in enumerate(bricks):
            if b[4] != k:
                continue
            nx, ny = P.footprint(b[1], b[5])
            ax.add_patch(Rectangle((b[3], b[2]), ny, nx, facecolor=col[b[0]], edgecolor="k", lw=1.2))
            ax.text(b[3] + ny / 2, b[2] + nx / 2, "%d\n%s" % (n, b[1]), ha="center", va="center", fontsize=7)
        if k:
            for b in bricks:
                if b[4] == k - 1:
                    nx, ny = P.footprint(b[1], b[5])
                    ax.add_patch(Rectangle((b[3], b[2]), ny, nx, fill=False, edgecolor="#888", lw=.8, ls=":"))
        ax.set_xlim(-.2, NJ + .2)
        ax.set_ylim(NI + .2, -.2)
        ax.set_aspect("equal")
        ax.set_title("layer %d" % k, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    save(fig, name)


layers_figure("layers_arch.png", arch, "Arch, layer by layer (dotted: the layer below). Numbers = build step.")
layers_figure("layers_pyramid.png", pyramid, "Pyramid, layer by layer (planner order before re-sequencing).")

fig = plt.figure(figsize=(4.2, 3.6))
ax = fig.add_subplot(projection="3d")
draw_bricks3d(ax, P.sequence(P.STRUCTURES["S3"]))
ax.set_title("S3 corbelled bridge (direct input, 16 bricks)")
ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
save(fig, "s3_bricks.png")


# ---- 3. support analysis ------------------------------------------------------
def plan(run, arm):
    return json.loads((run / "plans" / ("%s.json" % arm)).read_text())


fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [11, 16]})
for ax, arm, title in zip(axs, "AB", ("Arch (arm A)", "S3 (arm B)")):
    seq = plan(RUN, arm)["sequence"]
    u0 = [s["predicted_util_unbraced"] or 0 for s in seq]
    ub = [s["brace"]["predicted_util_braced"] if s["requires_brace"] else np.nan for s in seq]
    x = np.arange(len(seq))
    ax.bar(x - .2, u0, .4, color=["#d62728" if s["requires_brace"] else "#9ecae1" for s in seq],
           label="no support")
    ax.bar(x + .2, ub, .4, color="#2ca02c", label="with planned support")
    ax.axhline(1 / 3, color="k", ls="--", lw=1)
    ax.text(-.6, 1 / 3 + .03, "support threshold (u x 3 >= 1)", ha="left", fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("build step")
    ax.set_title(title)
    ax.set_ylim(0, max(max(u0) * 1.12, .6))
axs[0].set_ylabel("predicted joint utilisation u\n(load / strength, worst joint)")
from matplotlib.patches import Patch  # noqa: E402
axs[1].legend(handles=[Patch(color="#9ecae1", label="no support needed"),
                       Patch(color="#d62728", label="unsupported (step needs support)"),
                       Patch(color="#2ca02c", label="with the planned support")], loc="upper left", frameon=False)
save(fig, "utilisation.png")


def side_elevation(ax, bricks, step, brace):
    """S3 seen from the side (y-z): placed bricks, the new brick, the support pad."""
    order = P.sequence(bricks)
    for n, b in enumerate(order[:step + 1]):
        nx, ny = P.footprint(b[1], b[5])
        new = n == step
        ax.add_patch(Rectangle((b[3], b[4]), ny, 1, facecolor="#ffbf00" if new else "#c6dbef",
                               edgecolor="k", lw=1.6 if new else .8, hatch="//" if new else None))
        ax.text(b[3] + ny / 2, b[4] + .5, str(n), ha="center", va="center", fontsize=7)
    y0 = brace["origin"][1]
    py, pz = (brace["brace_pose"][1] - y0) / P.PITCH, (brace["brace_pose"][2] - brace["origin"][2]) / P.BRICK_H
    h = .0085 / P.PITCH
    ax.add_patch(Rectangle((py - h, pz - .0085 / P.BRICK_H), 2 * h, 2 * .0085 / P.BRICK_H,
                           facecolor="#2ca02c", alpha=.55, edgecolor="#1b5e20", lw=1.5))
    ax.annotate("helper's finger pads", (py, pz), (5, 2.5), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->"))
    tgt = order[step]
    nx, ny = P.footprint(tgt[1], tgt[5])
    ax.annotate("press %.0f N" % brace["press_nominal_N"], (tgt[3] + ny / 2, tgt[4] + 1),
                (tgt[3] + ny / 2, tgt[4] + 2.3), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#d62728"))
    ax.set_xlim(-1, 12)
    ax.set_ylim(-.3, 10.5)
    ax.set_aspect(P.BRICK_H / P.PITCH)
    ax.set_xlabel("y (studs)")
    ax.set_ylabel("layer")


fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
pB = plan(RUN, "B")
bricksB = [(b["id"], b["type"], *b["grid_pos"], b["yaw_index"]) for b in pB["bricks"]]
for ax, step in zip(axs, (11, 13, 14)):
    s = pB["sequence"][step]
    brace = dict(s["brace"], origin=pB["voxel_origin"])
    side_elevation(ax, bricksB, step, brace)
    ax.set_title("S3 step %d: u %.2f -> %.2f with support" % (
        step, s["predicted_util_unbraced"], s["brace"]["predicted_util_braced"]), fontsize=9)
save(fig, "s3_supports.png")


# ---- 4. task execution: timeline, accuracy --------------------------------------
def timeline(run, name, labels):
    ev = json.loads((run / "events.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 2.6))
    rows = {"A": 1, "B": 0}
    start = {}
    for e in ev:
        if e["event"] == "step_started" or e["event"] == "support_granted":
            start[e["arm"]] = (e["time"], e["step"], e["event"] == "support_granted")
        elif e["event"] == "step_complete":
            t0, step, sup = start.pop(e["arm"])
            ax.barh(rows[e["arm"]], e["time"] - t0, left=t0, height=.55,
                    color="#d62728" if sup else ("#1f77b4" if e["arm"] == "A" else "#ff7f0e"),
                    edgecolor="white")
            ax.text((t0 + e["time"]) / 2, rows[e["arm"]], str(step), ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")
    held = {}
    for e in ev:
        if e["event"] == "support_held":
            held[e["arm"]] = e["time"]
        if e["event"] == "support_released":
            helper = e["helper"]
            ax.barh(rows[helper] + .38, e["time"] - held[e["arm"]], left=held[e["arm"]], height=.16,
                    color="#2ca02c")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("simulation time (s)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1f77b4", label="A own step"), Patch(color="#ff7f0e", label="B own step"),
                       Patch(color="#d62728", label="supported step (other arm helps)"),
                       Patch(color="#2ca02c", label="helper holding the structure")],
              ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.32), frameon=False, fontsize=8)
    ax.set_ylim(-.5, 1.6)
    save(fig, name)


timeline(RUN, "timeline_arch_s3.png", ["B: S3", "A: arch"])
timeline(RUN_PYR, "timeline_pyramid_s3.png", ["B: S3", "A: pyramid"])

fig, axs = plt.subplots(1, 2, figsize=(11, 3.1), sharey=False)
for run, mark, lab in ((RUN, "o", "arch + S3 (012)"), (RUN_PYR, "s", "pyramid + S3 (006)")):
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    for k, arm in enumerate("AB"):
        r = [e for e in rows if e["structure"].startswith(arm)]
        axs[0].scatter([e["final_error_mm"] for e in r], [e["dz_mm"] for e in r], marker=mark, s=22,
                       alpha=.75, label="%s, arm %s" % (lab, arm))
axs[0].add_patch(Rectangle((0, -1.5), 2.0, 3.0, fill=False, ls="--", lw=1.2))
axs[0].text(1.95, 1.35, "snap gate", ha="right", fontsize=8)
axs[0].set_xlim(0, 2.3)
axs[0].set_ylim(-1.8, 1.8)
axs[0].set_xlabel("sideways error (mm)")
axs[0].set_ylabel("height error (mm)")
axs[0].legend(fontsize=7, frameon=False, loc="lower right")
axs[0].set_title("Every placed brick vs. the snap gate")
for run, lab in ((RUN, "arch + S3"), (RUN_PYR, "pyramid + S3")):
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    axs[1].hist([e["tilt_deg"] for e in rows], bins=np.linspace(0, 5, 21), alpha=.6, label=lab)
axs[1].axvline(5, color="k", ls="--")
axs[1].set_xlabel("tilt at snap (deg); gate 5 deg")
axs[1].set_ylabel("bricks")
axs[1].legend(frameon=False, fontsize=8)
axs[1].set_title("Tilt of every placed brick")
save(fig, "snap_accuracy.png")


# ---- 5. numbers the report quotes ----------------------------------------------
facts = {}
for run in (RUN, RUN_PYR):
    s = json.loads((run / "summary.json").read_text())
    v = json.loads((run / "verification.json").read_text())
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    facts[run.name] = dict(
        verifier_ok=v["ok"], full_build=v["full_build"], frames=v["frames"],
        parallel_frames=v["recorded_parallel_frames"], sim_time_s=max(e["sim_time_s"] for e in rows),
        placed={a: "%d/%d" % (s["builds"][a]["placed"], s["builds"][a]["total"]) for a in "AB"},
        plans=s["motion"]["plans"], plan_failures=s["motion"]["plan_failures"],
        min_clearance_mm=round(s["motion"]["min_clearance_m"] * 1000, 2),
        max_lateral_mm=max(e["final_error_mm"] for e in rows), max_abs_dz_mm=max(abs(e["dz_mm"]) for e in rows),
        max_tilt_deg=max(e["tilt_deg"] for e in rows),
        contacts={k: c["count"] for k, c in s["contacts"].items() if isinstance(c, dict) and "count" in c},
        supports=[(a, x["step"], x["realised_grip_N"]) for a in "AB" for x in s["builds"][a]["supports"]],
        brace_search=[(a, x["step"], x["accepted"], x.get("error", "")) for a in "AB"
                      for x in s["builds"][a]["brace_search"]])


# ---- 6. stud-fit (real insertion force) progress, full pyramid + S3 runs ----------
CAUSE = {"gate_miss": "brick missed the snap check", "motion_plan_failed": "no collision-free path",
         "support_not_established": "support grip not established", "support_lost": "support grip lost",
         "wait_timeout": "scheduler wait timed out", "command_clearance": "clearance guard stopped an arm",
         "joint_break": "a stud joint broke"}
rows = []
for run in sorted((ROOT / "Kean" / "results").glob("dual_pyramid_s3_studfit_0*")):
    s, vf = run / "summary.json", run / "verification.json"
    if not s.is_file():
        continue
    s, v = json.loads(s.read_text()), json.loads(vf.read_text()) if vf.is_file() else {}
    fail = s.get("failure") or ""
    key = fail.split(":")[1] if fail.startswith(("A:", "B:")) else fail.split(":")[0]
    if v.get("ok"):
        label = "verifier PASS"
    elif fail:
        label = CAUSE.get(key, fail)
    else:
        label = "all placed; verifier: %s" % ", ".join(sorted({e.split(": ")[-1] for e in v.get("errors", [])
                                                               if e.startswith("contact monitor")}))
    b = s["builds"]
    rows.append((run.name[-3:], b["A"]["placed"], b["B"]["placed"], label))
facts["studfit_runs"] = rows
fig, ax = plt.subplots(figsize=(11, .3 * len(rows) + 1.0))
for n, (name, a, bb, label) in enumerate(rows):
    ax.barh(n, a, color="#1f77b4", height=.6)
    ax.barh(n, bb, left=a, color="#ff7f0e", height=.6)
    ax.text(28.5, n, label, va="center", fontsize=8)
ax.axvline(27, color="k", ls="--", lw=1)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(["run " + r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlim(0, 46)
ax.set_xticks([0, 5, 10, 15, 20, 27])
ax.set_xlabel("bricks placed, out of 27 (blue: pyramid, 11; orange: S3, 16)")
ax.set_title("Full pyramid + S3 builds with real stud insertion force (--stud-fit)", fontsize=10)
save(fig, "studfit_progress.png")

(OUT / "facts.json").write_text(json.dumps(facts, indent=1))
print(json.dumps(facts, indent=1))
