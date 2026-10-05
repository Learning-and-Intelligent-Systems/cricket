#!/bin/bash
# The Rainbow candidate cover (cricket input URDF) from the current cricket input.
#   make_candidate.sh OUT.urdf
PY=/Users/tlp/Documents/GitHub/NQR/.pixi/envs/nhpn-ik/bin/python
cd "$(dirname "$0")"
DMAP="ee_finger_r1=6 ee_finger_r2=6 ee_finger_l1=6 ee_finger_l2=6 \
ee_right=8 ee_left=8 FT_sensor_R=8 FT_sensor_L=8 \
link_right_arm_5=8 link_right_arm_6=8 link_left_arm_5=8 link_left_arm_6=8 \
link_right_arm_4=10 link_left_arm_4=10 \
link_right_arm_0=15 link_right_arm_1=15 link_left_arm_0=15 link_left_arm_1=15 \
link_torso_0=20 link_torso_1=20 link_torso_2=20 link_torso_3=20 link_torso_4=15 link_torso_5=15 \
base=15 wheel_r=10 wheel_l=10 link_head_0=15 link_head_1=15 link_head_2=15"
# default delta 12 mm (arm_2, arm_3, lidar), radius pad 1 mm, cover targets every 3 mm
$PY respherize.py template_plus.urdf "$1" 12 $DMAP --pad=1 --spacing=3
