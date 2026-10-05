"""Second cricket pass: add the per-link sphere CLUSTERS (indexed like the
first pass's link_names, sphere indices as cricket numbers them) to the
configuration, for templates/{fk,ccfk}_template_h.hh.
    make_cricket_config.py PASS1_OUT.json URDF NAME EE OUT.json [CLUSTER_SIZE]"""
import json
import math
import sys

import numpy as np

from clusters import link_clusters
from rb_common import link_spheres


def main(p1, urdf, name, ee, out, size=8, urdf_rel=None, srdf_rel=None):
    d = json.load(open(p1))
    names = d["link_names"]
    S = link_spheres(urdf, merge_parts=False)
    clusters = [[] for _ in names]
    n_cl = 0
    for f in d["links_with_geometry"]:
        idx = d["per_link_spheres"][f]
        sp = S[names[f]]
        assert len(idx) == len(sp), names[f]
        C = np.array([c for c, _ in sp])
        R = np.array([r for _, r in sp])
        for members, anchor, radius in link_clusters(C, R, size):
            # radius rounded UP (the screen must contain every member)
            r = math.ceil((radius + 2e-6) * 1e6) / 1e6
            assert (np.linalg.norm(C[members] - C[anchor], axis=1) + R[members]).max() <= r
            clusters[f].append(dict(anchor=int(idx[anchor]), radius=r, spheres=[int(idx[m]) for m in members]))
            n_cl += 1
    cfg = {
        "name": name,
        "urdf": urdf_rel,
        "srdf": srdf_rel,
        "end_effector": ee,
        "resolution": 32,
        "template": "templates/fk_template_h.hh",
        "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template_h.hh"}],
        "output": f"{name.lower()}_fk.hh",
        "clusters": clusters,
    }
    json.dump(cfg, open(out, "w"), indent=1)
    print(f"{out}: {n_cl} clusters over {len(d['links_with_geometry'])} links")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], a[4], int(a[5]) if len(a) > 5 else 8, a[6] if len(a) > 6 else None, a[7] if len(a) > 7 else None)
