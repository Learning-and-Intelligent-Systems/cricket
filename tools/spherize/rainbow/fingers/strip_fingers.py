"""Drop the collision spheres of the given links from a spherized URDF (the
links and joints stay: kinematics and frame numbering are the module's own).
    strip_fingers.py IN OUT LINK [LINK ...]"""
import re
import sys

src, dst, links = sys.argv[1], sys.argv[2], set(sys.argv[3:])
text = open(src).read()
out, n = [], 0
for block in re.split(r"(?=<link name=)", text):
    m = re.match(r'<link name="([^"]+)"', block)
    if m and m.group(1) in links:
        block, k = re.subn(r"\s*<collision>.*?</collision>", "", block, flags=re.S)
        n += k
        links.discard(m.group(1))
    out.append(block)
assert not links, f"links not found: {links}"
open(dst, "w").write("".join(out))
print(f"{dst}: {n} spheres dropped")
