# Tight, conservative re-spherization (cricket inputs)

Tools that rebuild a robot's spherized URDF from its MASTER URDF, the
planner's own geometry, whose collision meshes every consumer hulls.
Each link gets a cover of its convex hull that:

- contains the hull: surface, edges, vertices and an interior grid, checked on
  an independent dense sample, with a 2 mm radius pad for the gaps between
  cover samples;
- stays within `delta` of it: every sphere lies inside the hull's
  delta-offset polytope, and above a floor for a mounted base link.

Run with a Python that has `qaa`, `trimesh`, `coal`, `lxml` (NQR's
`nhpn-ik` env); `SPHERIZE_MASTER` overrides the master URDF.

| script | does |
|---|---|
| `respherize.py TEMPLATE OUT DELTA_MM [link=mm ...]` | the cover, per link, into the template's links |
| `split_urdf.py IN.urdf IN.srdf OUT.urdf OUT.srdf link=k ...` | split big links into fixed `L__sN` sub-links (each gets its own bounding sphere in the generated kernel) and carry the SRDF over |
| `fit_report.py URDF` | per link: hull coverage and protrusion |
| `union_cover.py MASTER URDF REF_LINK [N]` | world-frame union coverage at random confs |
| `selfpair_audit.py URDF SRDF [N]` | per pair: sphere vs exact (Coal) hits, spurious and missed |
| `self_cost.py` / `split_cost.py URDF SRDF [link=k ...]` | expected self-collision sphere tests per conf of the generated kernel |
| `vamp_bench.py make/run/exact/score` | a built kernel's verdicts vs exact, and its speed |
| `mix_urdf.py NEW OLD OUT link ...` | a hybrid: NEW with the listed links' spheres taken from OLD (bisecting a regression) |

## Droid, 2026-10-02 (`resources/droid/`)

```
respherize.py droid_spherized_53.urdf cand.urdf 10 panda_link0=15 panda_link5=6 panda_link7=6 wrist_camera_link=8
# SRDF: the 53-sphere model's pairs, plus link5-wrist_camera and link2-link6
#       enabled (both collide within the joint limits); link5-link7 stays enabled
split_urdf.py cand.urdf cand.srdf droid_spherized.urdf droid_partial.srdf \
    panda_link5=8 panda_link7=3 wrist_camera_link=4 panda_link0=3 panda_link1=3 panda_link2=3
```

458 spheres (was 53), over the master's full joint ranges. `selfpair_audit` over
4000 random confs: no enabled pair misses a real (hull) collision; spheres
over-report link5-link7 in 3.1% of confs and link5-wrist_camera in 1.9% (the old
model: 35% for link5-link7, and it did not check link5-wrist_camera, which
really collides in 2% of confs and in 37% of the droid's canned table looks).

Base spheres stay at or above the base plane z = 0 and cover the hull from z = 12 mm
up. A first version let them reach z = -0.015, which collided at EVERY conf with the
PO worlds' permanent `floor` body (top z = -0.013): the exact checker skips bodies
named floor, VAMP does not, so every PO motion plan failed while FO and the IK census
looked fine (vamp 1ba2916, reverted). In a table scene against the exact hulls, at 20k
random confs: false-invalid 32% -> 5.2% and false-valid 375 -> 0. Node 218 of a
harder run: VAMP-only rejections of red's grasp branches 455 -> 58.

## Droid, 2026-10-03: the camera with its bracket clamp

The master's camera mesh grew to cover the DROID bracket's side clamp, which stuck
out 13 mm past the old box (QAA `wrist_camera_box.stl`: the hull of the camera
box and a clamp box). Only `wrist_camera_link` was re-covered; every other
link keeps its spheres:

```
respherize.py droid_spherized_full.urdf resph.urdf 10 panda_link0=15 panda_link5=6 panda_link7=6 wrist_camera_link=8
mix_urdf.py droid_spherized_full.urdf resph.urdf droid_spherized_full2.urdf wrist_camera_link
split_urdf.py droid_spherized_full2.urdf cand.srdf droid_spherized.urdf droid_partial.srdf \
    panda_link5=8 panda_link7=3 wrist_camera_link=4 panda_link0=3 panda_link1=3 panda_link2=3
```

(`droid_spherized_full.urdf` is the 10-02 cover before the split, with the floored
base.) Still 458 spheres (the camera keeps 57), the SRDF unchanged. Against the new
hull, the old camera spheres left 8.4% of it uncovered by up to 9 mm; the new ones
cover all of it. `selfpair_audit`, 4000 confs: no enabled pair misses; link5-wrist_camera
really collides in 2.6% of confs (2.2% with the old box) and the spheres over-report
it in 2.2%.
