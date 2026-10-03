"""Hybrid sphere model: the NEW spherized URDF with the listed links' spheres
replaced by the OLD model's (and their split sub-links emptied).
    mix_urdf.py NEW.urdf OLD.urdf OUT.urdf link ..."""
import copy
import sys

from lxml import etree

new_u, old_u, out = sys.argv[1:4]
links = set(sys.argv[4:])
tn = etree.parse(new_u)
to = etree.parse(old_u)
old = {l.get("name"): l for l in to.getroot().findall("link")}
for link in tn.getroot().findall("link"):
    name = link.get("name")
    base = name.split("__s")[0]
    if base not in links:
        continue
    for c in link.findall("collision"):
        link.remove(c)
    if name == base:
        for c in old[base].findall("collision"):
            link.append(copy.deepcopy(c))
tn.write(out, pretty_print=True, xml_declaration=True, encoding="utf-8")
print("restored old spheres on", sorted(links))
