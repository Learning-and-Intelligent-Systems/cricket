"""Split big links of a spherized URDF into fixed sub-links (cricket gives every
link its own bounding sphere, so the per-pair screen prunes finer), and carry
the SRDF over: a disabled pair stays disabled for every part combination, and
the parts of one link never check each other.
    split_urdf.py IN.urdf IN.srdf OUT.urdf OUT.srdf link=k ..."""
import sys

import numpy as np
from lxml import etree

src_u, src_s, out_u, out_s = sys.argv[1:5]
split = {a.split("=")[0]: int(a.split("=")[1]) for a in sys.argv[5:]}
tree = etree.parse(src_u)
root = tree.getroot()
parts = {}
for link in root.findall("link"):
    name = link.get("name")
    cols = link.findall("collision")
    parts[name] = [name]
    k = split.get(name, 1)
    if k <= 1 or len(cols) < 2 * k:
        continue
    C = np.array([[float(v) for v in c.find("origin").get("xyz").split()] for c in cols])
    u = np.linalg.svd(C - C.mean(0))[2][0]
    order = np.argsort((C - C.mean(0)) @ u)
    chunks = np.array_split(order, k)
    for i, chunk in enumerate(chunks[1:], start=1):
        sub = f"{name}__s{i}"
        parts[name].append(sub)
        nl = etree.SubElement(root, "link", name=sub)
        for j in chunk:
            link.remove(cols[j])
            nl.append(cols[j])
        jt = etree.SubElement(root, "joint", name=f"{sub}_joint", type="fixed")
        etree.SubElement(jt, "parent", link=name)
        etree.SubElement(jt, "child", link=sub)
        etree.SubElement(jt, "origin", xyz="0 0 0", rpy="0 0 0")
tree.write(out_u, pretty_print=True, xml_declaration=True, encoding="utf-8")

s = etree.parse(src_s)
sr = s.getroot()
dis = [(d.get("link1"), d.get("link2")) for d in sr.findall("disable_collisions")]
for d in sr.findall("disable_collisions"):
    sr.remove(d)
seen = set()
def add(a, b, why):
    key = frozenset((a, b))
    if a == b or key in seen:
        return
    seen.add(key)
    e = etree.SubElement(sr, "disable_collisions", link1=a, link2=b, reason=why)
for a, b in dis:
    for pa in parts.get(a, [a]):
        for pb in parts.get(b, [b]):
            add(pa, pb, "Inherited")
for name, ps in parts.items():
    for i, a in enumerate(ps):
        for b in ps[i + 1:]:
            add(a, b, "Same link")
s.write(out_s, pretty_print=True, xml_declaration=True, encoding="utf-8")
print({k: len(v) for k, v in parts.items() if len(v) > 1}, "disabled pairs", len(seen))
