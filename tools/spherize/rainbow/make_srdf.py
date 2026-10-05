"""The Rainbow VAMP SRDF, derived from the master's hulls.

Checked: every link pair that can collide within VAMP's 23 DOF, except the
documented near-contact pairs below. Disabled, by reason:
  Derived   qaa.robot_description.derived(PROFILE).self_collision_exclusions
            (joined by a joint / one rigid group / always overlapping /
            closed linkage): the pairs no simulator checks either.
  Fixed     both links rigid in VAMP's model (the head's pan/tilt and the
            fingers are baked), so the pair's distance is a constant > 0.
  Never     no collision within the master limits: the exact hulls (Coal)
            never touched in the random census, and minimising their distance
            over the joints between the two links (pair_minimize.py) stays
            > 0. A pair counts as collidable if its left/right mirror is.
  Contact   the hulls are a few mm apart at the robot's own rest postures, so
            any conservative sphere cover overlaps there; checking the pair
            would reject the stow / home confs (see NEAR_CONTACT).
    make_srdf.py SPHERIZED.urdf OUT.srdf CENSUS.npz [MIN.json ...]"""
import itertools
import json
import sys

import numpy as np

# hulls 5.1 mm apart at the default (stow) conf, 10.8 mm at home; collide in
# 1.6% of random confs (up to 20 mm deep). Neither the old VAMP SRDF nor HPN's
# exact self-check (_SELF_COLLISION_PAIRS) checks them.
NEAR_CONTACT = {
    frozenset(("link_torso_5", "link_right_arm_1")): "hulls 5.1 mm apart at the stow conf",
    frozenset(("link_torso_5", "link_left_arm_1")): "hulls 5.1 mm apart at the stow conf",
}
RIGID_GROUPS = [  # links rigidly attached in VAMP's model
    {"base", "wheel_r", "wheel_l", "lidar_mid360"},
    {"link_torso_5", "link_head_0", "link_head_1", "link_head_2", "rby1_camera_mount"},
    {"link_right_arm_6", "FT_sensor_R", "ee_right", "ee_finger_r1", "ee_finger_r2", "link_right_arm_camera"},
    {"link_left_arm_6", "FT_sensor_L", "ee_left", "ee_finger_l1", "ee_finger_l2", "link_left_arm_camera"},
]


def mirror(name):
    for a, b in (("right", "left"), ("_R", "_L"), ("finger_r", "finger_l"), ("wheel_r", "wheel_l")):
        if a in name:
            return name.replace(a, b)
        if b in name:
            return name.replace(b, a)
    return name


def main(sph_urdf, out, census, *mins):
    from qaa.robot_description import derived

    from rb_common import PROFILE

    from rb_common import link_spheres

    links = [l for l, s in link_spheres(sph_urdf, merge_parts=True).items() if s]
    d = np.load(census, allow_pickle=True)
    pairs = [frozenset(map(str, p)) for p in d["pairs"]]
    dmin = dict(zip(pairs, d["D"].min(axis=0).astype(float)))
    override = {}
    for m in mins:
        for k, v in json.load(open(m)).items():
            p = frozenset(k.split("|"))
            if "census_hit" in v:  # camera_pairs.py: against the SWEPT camera, replaces the static evidence
                override[p] = v["min"]
            else:
                dmin[p] = min(dmin.get(p, np.inf), v["min"])
    dmin.update(override)
    ex = {frozenset(p) for p in derived(PROFILE).self_collision_exclusions}
    rows = []
    for a, b in itertools.combinations(links, 2):
        p = frozenset((a, b))
        pm = frozenset((mirror(a), mirror(b)))
        if p in ex:
            rows.append((a, b, "Derived", None))
            continue
        if any(p <= g for g in RIGID_GROUPS):
            rows.append((a, b, "Fixed", dmin.get(p)))
            continue
        if p in NEAR_CONTACT:
            rows.append((a, b, "Contact", dmin.get(p)))
            continue
        m = min(dmin.get(p, np.inf), dmin.get(pm, np.inf))
        if not np.isfinite(m):
            raise RuntimeError(f"no evidence for {a} -- {b}")
        rows.append((a, b, "Never" if m > 0 else "CHECK", m))
    lines = ['<?xml version="1.0" encoding="utf-8"?>', f'<robot name="{PROFILE}">']
    n = {}
    for r in rows:
        n[r[2]] = n.get(r[2], 0) + 1
    lines.append(
        "  <!-- Rainbow VAMP self-collision pairs (spheres_rainbow_1004/make_srdf.py), "
        f"{len(links)} links, {len(rows)} pairs: {n.get('CHECK', 0)} checked; disabled: "
        + ", ".join(f"{k} {v}" for k, v in sorted(n.items()) if k != "CHECK")
        + ". Derived = qaa derived('rainbow').self_collision_exclusions; Fixed = rigid in "
        "VAMP's model (head pan/tilt baked); Never = the exact hulls never touch within "
        "the master limits (random census + distance minimisation); Contact = hulls a few "
        "mm apart at the stow posture. -->"
    )
    for a, b, why, m in rows:
        if why == "CHECK":
            continue
        reason = {"Derived": "Adjacent", "Fixed": "Fixed", "Never": "Never", "Contact": "Contact"}[why]
        lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{reason}"/>')
    lines.append("</robot>")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{out}: {len(links)} links, {len(rows)} pairs -> checked {n.get('CHECK', 0)}; disabled",
          {k: v for k, v in n.items() if k != "CHECK"})
    near = sorted((m, a, b, w) for a, b, w, m in rows if w in ("Never", "Fixed") and m is not None and m < 0.02)
    print("disabled pairs with min exact distance < 20 mm:", [(a, b, w, round(1000 * m, 1)) for m, a, b, w in near])
    return rows


if __name__ == "__main__":
    main(*sys.argv[1:])
