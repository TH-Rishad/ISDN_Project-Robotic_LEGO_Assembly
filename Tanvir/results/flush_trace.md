# FLUSH sockets: why the scripted wriggle fails, and the pad shift

Traced with `Tanvir/flush_trace.py` (per-tick contacts) and `Tanvir/flush_padshift.py` (paired comparison). ScriptedWriggle (theta = 0), cheap sensing, DR on, no brace, env seed 10000.

## Trace

Context `fam_FLUSH_0_100001`: 1x6 target (7.8 mm wide), 2x4 neighbour on the -y face, free pinch axis x, aim error 0.26 mm.

| step | phase | brick above seat | hand force | contacts |
|---|---|---|---|---|
| 27-108 | search | 7.0-7.5 mm (never lower) | 11-14 N | `hand/padn` on `nb0` studs (3-4 contacts every tick) |
| 111-123 | search | 5.2-5.7 mm | 10-45 N | `hand/padn` on `nb0` (1-2 contacts) |
| 126-144 | press | 5.1-5.3 mm | ramp to 71.5 N | `hand/padn` on `nb0` |
| 147-200 | search | 5.2-5.7 mm, brick 1.9-3.9 mm off its seat | 9-12 N | `hand/padn` on `nb0`; timeout |

Context `fam_FLUSH_1_100002`: 2x2 target on layer 1, 2x4 neighbour `nb0` on -y, pinch axis x. The descent stopped 5.3 mm above the seat with `hand/padn` on `nb0` studs (4 contacts); at step 29 the neighbour's joint to the layer below (`nb0`/`L000`) broke at a peak hand force of 12.6 N (structure utilisation 0.20 at the last sample before the break).

In both, the finger pad on the -x end of the brick is the body touching the neighbour, not the brick. The neighbour-to-target contacts that appear (2 at steps 93 and 102) are brief.

## Cause

The pads are 17.5 mm wide (`FINGER_W`, the v4 cell's grasp) and centred on the brick. A 1-wide brick is 7.8 mm wide and a 2-wide one 15.8 mm, so the pads overhang the brick by 4.85 mm and 0.85 mm per side. With a flush neighbour on one side of the brick, the overhang lies over the neighbour, and the pad tip (4.1 mm above the brick bottom) lands on the neighbour's studs. The audit's "free pinch axis" looks only at the cells at the brick's ends and ignores the pad width. So the descent is blocked a few mm above the seat, and the scripted controller (which only releases when the socket is marked finger-blocked) keeps spiralling and pressing with the pad on the neighbour's studs: timeouts, 70 N presses and joint breaks of the neighbour.

This is a grasp-geometry effect that is physical for a 17.5 mm pad, not a controller bug; the controller does not choose where the pads sit. It is in the twin as an env choice (pad position), not in the terminations, budgets, gates or joint strengths, none of which are changed.

## Fix

`SocketInsertionEnv(pad_shift=True)` (default False, so all older results are unchanged). When exactly one of the two faces perpendicular to the pinch axis touches a neighbour, the pads are placed off-centre along that face, away from the neighbour, with the pad edge 0.2 mm inside the brick's face (a grasp-point choice from the plan). Nothing else changes. Tests: `tests/test_skill.py` (pad positions; `fam_FLUSH_1_100002` goes joint_break to seated).

## Before / after (scripted, one seed, no brace)

FLUSH family sockets, 6 train + 6 validation contexts, levels 0 and 2 (24 episodes):

| | seated | outcomes |
|---|---|---|
| pad centred (old) | 9/24 (level 0: 5/12, level 2: 4/12) | joint break 7, timeout 6, force violation 2 |
| pad shifted | **16/24** (level 0: 9/12, level 2: 7/12) | timeout 3, force violation 3, joint break 2 |

Seven episodes changed outcome, all to seated (4 from joint break, 1 from force violation, 2 from timeout); none got worse.

| context | brick | face | level | old | new |
|---|---|---|---|---|---|
| FLUSH train seed 2 | 2x4 | -x | 2 | timeout | seated |
| FLUSH train seed 4 | 2x3 | -y | 2 | joint break | seated |
| FLUSH train seed 5 | 2x4 | -x | 0 | force violation | seated |
| `fam_FLUSH_0_100001` | 1x6 | -y | 0 | timeout | seated |
| `fam_FLUSH_1_100002` | 2x2 | -y | 0 | joint break | seated |
| `fam_FLUSH_1_100002` | 2x2 | -y | 2 | joint break | seated |
| `fam_FLUSH_*_100003` | 1x2 | -x | 0 | joint break | seated |

The repo's two pyramid FLUSH steps (3 seeds, levels 0 and 2, no brace):

| context | pad centred | pad shifted |
|---|---|---|
| `pyramid_b_006_exec` | 0/6 | **6/6** |
| `pyramid_b_008_exec` | 0/6 | **5/6** |

CORNER and SLOT (24 episodes each) are identical with and without the shift (10/24 and 7/24): their pinch axes are blocked or have neighbours on both sides, so the shift does not apply.

Caveat: one seed per episode and 24 episodes for the family table; the pyramid result (0/12 to 11/12) is the clearest. Results from before this change (scripted_cheap_v2, `skill_main`) were produced with the pad centred; the ES run (§6.11) uses the shift and compares against theta = 0 with the shift.
