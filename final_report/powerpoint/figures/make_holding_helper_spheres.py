"""Illustrate the helper's real FR3 collision spheres at a recorded support hold.

Uses Kean/results/dual_pyramid_s3_studfit_031 (recording.npz) at S3 step 13
support_held: helper = arm A, builder = arm B. Sphere centres/radii come from
Sakib/brickassembly/motion/fr3.yml (the full clearance model). Link poses are
Franka MDH FK + the URDF hand −π/4 flange offset and measured finger openings
(same shift rule as Kean/motion/clearance.py).
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "Kean" / "results" / "dual_pyramid_s3_studfit_031"
FR3_YML = ROOT / "Sakib" / "brickassembly" / "motion" / "fr3.yml"
OUT = Path(__file__).resolve().parent
FIGS = ROOT / "final_report" / "figures"
VIDS = ROOT / "final_report" / "videos"
TMP = OUT / "_tmp_frames"


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        for c in (
            Path.home() / "Library/Python/3.14/lib/python/site-packages/imageio_ffmpeg/binaries",
        ):
            hits = list(c.glob("ffmpeg*"))
            if hits:
                return str(hits[0])
        return "ffmpeg"

# dual_arm_sim.ARM_A / ARM_B
BASE = {
    "A": (np.array([-0.45, 0.0, 0.0]), 0.0),
    "B": (np.array([0.45, 0.0, 0.0]), math.pi),
}
FINGER_LOCK = 0.0075  # cuRobo locked opening (m); clearance.py shifts from this


# --- Franka MDH (libfranka / official) -----------------------------------------------------------------
# link i: modified DH; theta_i = q_i. After link7: flange Trans(z, 0.107), then hand Rz(-π/4).
_MDH = [  # (a, d, alpha)
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -math.pi / 2),
    (0.0, 0.316, math.pi / 2),
    (0.0825, 0.0, math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0, 0.0, math.pi / 2),
    (0.088, 0.0, math.pi / 2),
]
_FLANGE_D = 0.107
_HAND_RZ = -math.pi / 4


def _mdh(a, d, alpha, theta):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0, 0, 0, 1],
    ], float)


def link_poses(q7):
    """World-from-link 4x4 for fr3_link1..7, fr3_hand (base frame of the arm)."""
    T = np.eye(4)
    out = {}
    for i, ((a, d, alpha), qi) in enumerate(zip(_MDH, q7), start=1):
        T = T @ _mdh(a, d, alpha, float(qi))
        out[f"fr3_link{i}"] = T.copy()
    flange = T @ _mdh(0.0, _FLANGE_D, 0.0, 0.0)
    hand = flange @ _mdh(0.0, 0.0, 0.0, _HAND_RZ)
    out["fr3_hand"] = hand
    return out


def yaw_R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def from_base(arm, pts):
    origin, yaw = BASE[arm]
    return np.asarray(pts, float) @ yaw_R(yaw).T + origin


def load_spheres(yml=FR3_YML):
    cs = yaml.safe_load(yml.read_text())["robot_cfg"]["kinematics"]["collision_spheres"]
    return {link: [(np.array(s["center"], float), float(s["radius"])) for s in specs]
            for link, specs in cs.items()}


# panda_finger_joint origin on the hand (Franka URDF): 58.4 mm along hand +z, then ±y opening.
_FINGER_Z = 0.0584


def arm_world_spheres(arm, joints9, sphere_model, links=("fr3_hand", "fr3_leftfinger", "fr3_rightfinger",
                                                          "fr3_link6", "fr3_link7")):
    """(N,4) world xyzr for selected links. Finger openings applied like clearance.py."""
    q7 = np.asarray(joints9[:7], float)
    gL, gR = float(joints9[7]), float(joints9[8])
    poses = link_poses(q7)
    hand = poses["fr3_hand"]
    # Finger frames: URDF origin on hand +z, then prismatic ±y at the measured opening.
    left = hand.copy()
    left[:3, 3] = hand[:3, 3] + hand[:3, 2] * _FINGER_Z + hand[:3, 1] * gL
    right = hand.copy()
    right[:3, 3] = hand[:3, 3] + hand[:3, 2] * _FINGER_Z - hand[:3, 1] * gR
    poses["fr3_leftfinger"] = left
    poses["fr3_rightfinger"] = right

    rows = []
    for link in links:
        if link not in sphere_model or link not in poses:
            continue
        T = poses[link]
        R, p = T[:3, :3], T[:3, 3]
        for c_local, r in sphere_model[link]:
            c = R @ c_local + p
            rows.append([*c, r])
    sp = np.asarray(rows, float)
    sp[:, :3] = from_base(arm, sp[:, :3])
    return sp


def min_gap(a, b):
    d = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=2) - a[:, None, 3] - b[None, :, 3]
    return float(d.min())


def load_hold(step=13):
    z = np.load(RUN / "recording.npz", allow_pickle=True)
    events = json.loads((RUN / "events.json").read_text())
    hold = next(e for e in events if e["event"] == "support_held" and e["step"] == step)
    release = next(e for e in events if e["event"] == "support_released" and e["step"] == step
                   and e["time"] >= hold["time"])
    return z, hold, release


def closest_idx(z, hold, release, sphere_model):
    """Frame in [hold, release] where helper and builder distal spheres are closest."""
    links = ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_link7")
    best = (np.inf, int(np.searchsorted(z["sim_time"], hold["time"])))
    for t in np.arange(hold["time"], release["time"], 0.25):
        idx = int(np.searchsorted(z["sim_time"], t))
        q = z["q_exec"][idx]
        gap = min_gap(
            arm_world_spheres("A", q[:9], sphere_model, links=links),
            arm_world_spheres("B", q[9:18], sphere_model, links=links),
        )
        if gap < best[0]:
            best = (gap, idx)
    return best[1], best[0]


def bricks_at(z, idx, max_bricks=None):
    """List of (xyz, quat_xyzw, welded) for bricks present (approx by welded or any)."""
    ids = [str(x) for x in z["brick_ids"]]
    q = z["brick_q"][idx]
    welded = z["welded"][idx]
    out = []
    for i, bid in enumerate(ids):
        # Keep every brick that has a finite pose; filter tiny feeders by |z|.
        if not np.isfinite(q[i]).all():
            continue
        if q[i, 2] < -0.01:
            continue
        out.append((bid, q[i, :3].copy(), q[i, 3:7].copy(), bool(welded[i])))
    return out


# --- drawing ----------------------------------------------------------------------------------------------
def draw_spheres_3d(ax, spheres, color, alpha=0.35, stride=1):
    u = np.linspace(0, 2 * math.pi, 12)
    v = np.linspace(0, math.pi, 8)
    su, sv = np.meshgrid(u, v)
    for x, y, z, r in spheres[::stride]:
        xs = x + r * np.cos(su) * np.sin(sv)
        ys = y + r * np.sin(su) * np.sin(sv)
        zs = z + r * np.cos(sv)
        ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, shade=True)


def draw_brick_boxes(ax, bricks, size=(0.016, 0.016, 0.0096)):
    # Approximate each brick as an axis-aligned box at its pose (yaw from quat).
    sx, sy, sz = size
    for bid, p, quat, welded in bricks:
        if not bid.startswith("B_"):  # S3 is arm B's structure in this run
            continue
        R = Rot.from_quat(quat).as_matrix()
        # 8 corners of a 2x2x1 brick-ish box (fine for overview; true footprint varies)
        corners = np.array([[i, j, k] for i in (-sx / 2, sx / 2)
                            for j in (-sy / 2, sy / 2)
                            for k in (0, sz)])
        pts = (R @ corners.T).T + p
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], color="#93c5fd" if welded else "#fbbf24",
                        alpha=0.55, linewidth=0)


def figure_side_view(helper_sp, builder_sp, bricks, gap_mm, out_path):
    """Orthographic side view (looking along −x): Y horizontal, Z vertical — matches S3_SIDE stills."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=160)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")

    # Plate / floor cue from S3 pad (y ≈ −0.16)
    ax.add_patch(FancyBboxPatch((-0.28, -0.01), 0.24, 0.006, boxstyle="square,pad=0",
                                facecolor="#64748b", edgecolor="none", zorder=0))

    # Bricks (B_*) as rectangles in y–z
    for bid, p, quat, welded in bricks:
        if not bid.startswith("B_"):
            continue
        yaw = Rot.from_quat(quat).as_euler("xyz")[2]
        # footprint approx from yaw: treat as 2×2 stud
        ny, nz = 0.016, 0.0096
        if abs(math.cos(yaw)) < 0.7:  # roughly 90°
            ny = 0.032
        ax.add_patch(Rectangle((p[1] - ny / 2, p[2]), ny, nz,
                               facecolor="#93c5fd" if welded else "#fbbf24",
                               edgecolor="#1e293b", lw=0.6, alpha=0.9, zorder=2))

    # Draw spheres with r ≥ 2.5 mm (tiny mesh spheres clutter the plot; gap uses the full set).
    def visible(sp):
        return [row for row in sp if row[3] >= 0.0025]

    for x, y, z, r in visible(helper_sp):
        ax.add_patch(Circle((y, z), r, facecolor="#38bdf8", edgecolor="#0284c7",
                            lw=0.5, alpha=0.40, zorder=3))
    hlo = (helper_sp[:, :3] - helper_sp[:, 3:4]).min(axis=0)
    hhi = (helper_sp[:, :3] + helper_sp[:, 3:4]).max(axis=0)
    for x, y, z, r in visible(builder_sp):
        if y < hlo[1] - 0.12 or y > hhi[1] + 0.12 or z < hlo[2] - 0.12 or z > hhi[2] + 0.12:
            continue
        ax.add_patch(Circle((y, z), r, facecolor="#fb923c", edgecolor="#c2410c",
                            lw=0.35, alpha=0.28, zorder=3))

    # Closest pair
    d = np.linalg.norm(helper_sp[:, None, :3] - builder_sp[None, :, :3], axis=2) \
        - helper_sp[:, None, 3] - builder_sp[None, :, 3]
    i, j = np.unravel_index(np.argmin(d), d.shape)
    a, b = helper_sp[i], builder_sp[j]
    ax.plot([a[1], b[1]], [a[2], b[2]], color="#f87171", lw=1.8, zorder=5)
    ax.add_patch(Circle((a[1], a[2]), a[3] + 0.002, fill=False, edgecolor="#ef4444",
                        lw=1.6, linestyle="--", zorder=5))
    mid = ((a[1] + b[1]) / 2, (a[2] + b[2]) / 2 + 0.012)
    ax.text(mid[0], mid[1], f"gap {gap_mm:+.2f} mm\n(≤ 2 mm graze allowed)",
            color="#fecaca", fontsize=9, ha="center", va="bottom", zorder=6)

    ax.text(0.02, 0.98,
            "Next to a holding helper — real FR3 collision spheres\n"
            f"run dual_pyramid_s3_studfit_031 · S3 step 13 · helper A · full model (fr3.yml)",
            transform=ax.transAxes, color="#f8fafc", fontsize=11, va="top")
    ax.plot([], [], color="#38bdf8", label="helper spheres (A)")
    ax.plot([], [], color="#fb923c", label="builder spheres (B, nearby)")
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
    ax.set_xlabel("y (m)", color="#94a3b8")
    ax.set_ylabel("z (m)", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.set_aspect("equal")
    # Focus on the hold
    cy = float(np.median(helper_sp[:, 1]))
    cz = float(np.median(helper_sp[:, 2]))
    ax.set_xlim(cy - 0.14, cy + 0.14)
    ax.set_ylim(max(-0.02, cz - 0.10), cz + 0.14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out_path)


def figure_with_still(side_path, still_path, out_path):
    """Composite: real Newton still | measured sphere side-view."""
    left = Image.open(still_path).convert("RGB")
    right = Image.open(side_path).convert("RGB")
    h = 720
    left = left.resize((int(left.width * h / left.height), h), Image.Resampling.LANCZOS)
    right = right.resize((int(right.width * h / right.height), h), Image.Resampling.LANCZOS)
    gap = 16
    canvas = Image.new("RGB", (left.width + right.width + gap * 3, h + 64), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.text((gap, 16), "What the run did: Newton still  |  FR3 spheres at the recorded support pose",
              fill=(248, 250, 252), font=font)
    canvas.paste(left, (gap, 52))
    canvas.paste(right, (gap * 2 + left.width, 52))
    draw.text((gap, h + 36), "Newton playback (same step / time)", fill=(148, 163, 184), font=small)
    draw.text((gap * 2 + left.width, h + 36),
              "fr3.yml spheres @ recorded q_exec (FK)", fill=(148, 163, 184), font=small)
    canvas.save(out_path, quality=95)
    print("wrote", out_path)


def make_video(z, hold, release, sphere_model, out_path, fps=12):
    """Animate distal helper/builder spheres over the hold (through press)."""
    TMP.mkdir(parents=True, exist_ok=True)
    frames_dir = TMP / "real_spheres_frames"
    if frames_dir.exists():
        for p in frames_dir.glob("*.png"):
            p.unlink()
    else:
        frames_dir.mkdir()

    links = ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_link7")
    t0, t1 = hold["time"], min(release["time"], hold["time"] + 14.0)
    times = np.arange(t0, t1, 1.0 / fps)
    for i, t in enumerate(times):
        idx = min(int(np.searchsorted(z["sim_time"], t)), len(z["sim_time"]) - 1)
        q = z["q_exec"][idx]
        helper = arm_world_spheres("A", q[:9], sphere_model, links=links)
        builder = arm_world_spheres("B", q[9:18], sphere_model, links=links)
        bricks = bricks_at(z, idx)
        gap = min_gap(helper, builder) * 1000
        figure_side_view(helper, builder, bricks, gap, frames_dir / f"f_{i:04d}.png")
        if i % 12 == 0:
            print(f"video frame {i}/{len(times)} gap={gap:+.1f}mm", flush=True)

    pattern = str(frames_dir / "f_%04d.png")
    subprocess.run(
        [_ffmpeg(), "-y", "-framerate", str(fps), "-i", pattern,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print("wrote", out_path)


def ensure_still_frame():
    """Newton frame at S3 step 13 from the stud-fit montage (sim t ≈ 395 s)."""
    TMP.mkdir(parents=True, exist_ok=True)
    dest = TMP / "pyr_step13.jpg"
    if dest.exists():
        return dest
    src = VIDS / "dual_pyramid_s3_studfit_030_30s.mp4"
    if not src.exists():
        return FIGS / "sim_a_sup13_side.jpg"
    subprocess.run(
        [_ffmpeg(), "-y", "-ss", "20", "-i", str(src), "-frames:v", "1", str(dest)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return dest


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    ensure_still_frame()
    sphere_model = load_spheres(FR3_YML)
    z, hold, release = load_hold(13)
    idx, gap_m = closest_idx(z, hold, release, sphere_model)
    q = z["q_exec"][idx]
    # Distal links: what the builder actually works next to.
    links = ("fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_link7")
    helper = arm_world_spheres("A", q[:9], sphere_model, links=links)
    builder = arm_world_spheres("B", q[9:18], sphere_model, links=links)
    bricks = bricks_at(z, idx)
    gap = gap_m * 1000
    print(f"closest during hold t={z['sim_time'][idx]:.2f}s idx={idx}  "
          f"helper spheres={len(helper)}  builder={len(builder)}  gap={gap:+.2f} mm")
    print(f"  phases A={z['phase_names'][int(z['a_phase'][idx])]}  "
          f"B={z['phase_names'][int(z['b_phase'][idx])]}")

    side = OUT / "holding_helper_spheres_side.png"
    figure_side_view(helper, builder, bricks, gap, side)

    # Prefer a Newton frame from the same step/time as the sphere snapshot.
    still = TMP / "pyr_step13.jpg"
    if not still.exists():
        still = FIGS / "sim_a_sup13_side.jpg"
    composite = OUT / "holding_helper_spheres.jpg"
    if still.exists():
        figure_with_still(side, still, composite)
    else:
        composite.write_bytes(side.read_bytes())

    # Publish into report folders
    (FIGS / "holding_helper_spheres.jpg").write_bytes(composite.read_bytes())
    (FIGS / "holding_helper_spheres_side.png").write_bytes(side.read_bytes())

    # Drop the old hand-tuned still overlays
    for name in ("holding_helper_sup10_side.jpg", "holding_helper_sup11_side.jpg",
                 "holding_helper_sup13_side.jpg", "holding_helper_spheres_montage.jpg"):
        p = FIGS / name
        if p.exists():
            p.unlink()
            print("removed", p.name)
    for p in OUT.glob("holding_helper_sim_a_sup*_side.jpg"):
        p.unlink()
        print("removed", p.name)
    montage = OUT / "holding_helper_spheres_montage.jpg"
    if montage.exists():
        montage.unlink()

    # Short video of the real sphere geometry over the hold
    vid = OUT / "holding_helper_spheres.mp4"
    make_video(z, hold, release, sphere_model, vid, fps=12)  # 12 fps keeps render time sane
    (VIDS / "holding_helper_spheres.mp4").write_bytes(vid.read_bytes())
    print("done")


if __name__ == "__main__":
    main()
