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

## Droid, 2026-10-02 (`resources/droid/`)

```
respherize.py droid_spherized_53.urdf cand.urdf 10 panda_link0=15 panda_link5=6 panda_link7=6 wrist_camera_link=8
# set panda_joint6 lower = 0.95 (the master's limit since 2026-10-02)
# SRDF: the 53-sphere model's pairs + link5-wrist_camera and link2-link6 enabled
#       (real collisions), link5-link7 and link5-wrist_camera disabled again
#       (no contact for q6 >= 0.95)
split_urdf.py cand.urdf cand.srdf droid_spherized.urdf droid_partial.srdf \
    panda_link5=8 panda_link7=3 wrist_camera_link=4 panda_link0=3 panda_link1=3 panda_link2=3
```

452 spheres (was 53). Base spheres reach z >= -0.015, above the droid scenes'
table top at -0.02. In a table scene against the exact hulls, at 20k random
confs: false-invalid 32% -> 5.2% and false-valid 375 -> 0. Node 218 of a
harder run: VAMP-only rejections of red's grasp branches 455 -> 58.
