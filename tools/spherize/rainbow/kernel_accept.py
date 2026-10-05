"""The push_or_pick acceptance check on a BUILT kernel: the 72 edge confs
(18 sets x 4, exact-clear of the table by >= 6 mm) and the stowed default
conf at each of their 13 bases, in the env HPN builds for perm_only checks
(the table, a permanent body > 0.3 m, rides as its bounds cuboid with x/y
half-extents grown by COLLISION_DISTANCE_THR = 1 mm).
    kernel_accept.py VAMPDIR|- [THR_MM]"""
import pickle
import sys

import numpy as np

from scene_check import EDGE_PKL, TABLE, ExactModel, default_conf, vamp_coords


def main(vdir, thr=0.001):
    if vdir != "-":
        sys.path.insert(0, vdir)
    import vamp

    r = vamp.rainbow
    env = vamp.Environment()
    c, h = TABLE
    env.add_cuboid(vamp.Cuboid(list(map(float, c)), [0.0, 0.0, 0.0], list(map(float, h + np.array([thr, thr, 0.0])))))
    found = pickle.load(open(EDGE_PKL, "rb"))
    ex = ExactModel()
    ok = n = 0
    rej = []
    bases = {}
    for e, qs in found:
        for i, q in enumerate(qs):
            v = r.validate(vamp_coords(q).astype(np.float32), env)
            ok += v
            n += 1
            if not v:
                rej.append((e, i, min(ex.box_clearance(q, c, h).values())))
            bases.setdefault(tuple(np.round(np.asarray(q["base"], float), 3)), q)
    sok = 0
    srej = []
    for b, q in bases.items():
        dq = default_conf(q["base"])
        v = r.validate(vamp_coords(dq).astype(np.float32), env)
        sok += v
        if not v:
            xc = ex.box_clearance(dq, c, h)
            srej.append((b, round(1000 * min(xc.values()), 1), min(xc, key=xc.get)))
    print(f"{vamp.__file__}: edge confs accepted {ok}/{n}; stowed default conf accepted at {sok}/{len(bases)} bases")
    for e, i, m in rej:
        print(f"   edge REJECT {e} #{i}: exact clearance {1000 * m:.1f} mm")
    for b, m, l in srej:
        print(f"   stow REJECT base {b}: exact clearance {m} mm ({l})")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) / 1000 if len(sys.argv) > 2 else 0.001)
