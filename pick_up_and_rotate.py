'''
In terminal:
python -m venv robotics
source robotics/bin/activate
pip install -r requirements.txt

python pick_up_and_rotate.py
'''
import numpy as np
import genesis as gs
import torch
import sys
import matplotlib.pyplot as plt
import settings as s
from helper import (count_particles_in_cup, approach, grasp, ungrasp, lift, move_dist, rotate)

def main(): 
    pour_level = sys.argv[1] if len(sys.argv) > 1 else "medium"
    print("Pour level (high, medium, or low): ", pour_level)

    ########################## init and create a scene ##########################
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (3, -1, 1.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        show_viewer = True,
    )

    ########################## add entities ##########################
    liquid = scene.add_entity(
            material=gs.materials.PBD.Liquid(),
            morph=gs.morphs.Cylinder(
                height=s.LIQUID_HEIGHT,        # 12 cm tall
                radius=s.LIQUID_RADIUS,        # 3 cm radius
                pos=s.LIQUID1_START_POS,  # sitting on plane (z = height/2)),
            ),
            surface = gs.surfaces.Default(
                color    = (1.0, 0.4, 0.4),
                vis_mode = 'particle'
            )
    )

    liquid2 = scene.add_entity(
            material=gs.materials.PBD.Liquid(),
            morph=gs.morphs.Cylinder(
                height=s.LIQUID_HEIGHT,        # 12 cm tall
                radius=s.LIQUID_RADIUS,        # 3 cm radius
                pos=s.LIQUID2_START_POS,  # sitting on plane (z = height/2)),
            ),
            surface = gs.surfaces.Default(
                color    = (0.3, 0.3, 1.0),
                vis_mode = 'particle'
            )
    )

    rod = scene.add_entity(
            morph=gs.morphs.Cylinder(
                height=s.ROD_HEIGHT,
                radius=s.ROD_RADIUS,
                pos=s.ROD_START_POS,  # same height as the cups, but displaced to the right
            )
    )

    plane = scene.add_entity(gs.morphs.Plane())

    cup = scene.add_entity(
        gs.morphs.Mesh(file=s.CUP_FILE, pos=s.CUP_START_POS, scale=s.CUP_SCALE, euler=(90, 0, 0)),
    )

    cup2 = scene.add_entity(
        gs.morphs.Mesh(file=s.CUP_FILE, pos=s.CUP2_START_POS, scale=s.CUP2_SCALE, euler=(90, 0, 0)),
    )

    cup3 = scene.add_entity(
        gs.morphs.Mesh(file=s.CUP_FILE, pos=s.CUP3_START_POS, scale=s.CUP_SCALE, euler=(90, 0, 0)),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )

    # cam = scene.add_camera(
    #     model='pinhole',
    #     res=(320, 320),
    #     pos=CAM_POS,        # put camera above robot
    #     lookat=CUP_START_POS,        # look at the cup
    #     up=(0,0,1),
    #     fov=60,
    #     GUI=False,                  # if True: opens a window with the camera view
    #     near=0.05,
    #     far=5.0,
    # )

    scene.build()

    # --- control gains ----------------------------------------------------------
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    )

    ########################## EXECUTION PIPELINE ##########################
    approach(scene, franka, s.CUP_START_POS)
    grasp(scene, franka)
    lift(scene, franka, s.CUP_START_POS, s.LIFT_HEIGHT)
    move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    rotate(scene, franka, pour_level)
    move_dist(scene, franka, 0, -1.0 * s.X_OFFSET[pour_level])
    lift(scene, franka, cup.get_pos().cpu().numpy(), 0.08)
    ungrasp(scene, franka)
    move_dist(scene, franka, 1, 0.05)
    move_dist(scene, franka, 1, 0.1)
    move_dist(scene, franka, 0, -0.2)

    # approach(scene, franka, ROD_START_POS)
    # grasp(franka)
    # lift(franka, ROD_START_POS, LIFT_HEIGHT)
    # move_dist(franka, 0, x_offset[pour_level])
    # stir(franka)

    # approach(franka, CUP3_START_POS)
    # grasp(franka)
    # lift(franka, CUP3_START_POS, lift_height)
    # move_horizontally(franka, x_offset[pour_level]) # x_offset should depend on pour level + what cup it is 
    # rotate(franka, pour_level)
    # move_horizontally(franka, -x_offset[pour_level])
    # lift(franka, cup3.get_pos().cpu().numpy(), 0.08)

    in_cup, total = count_particles_in_cup(cup, liquid, s.CUP_HEIGHT, s.CUP_RADIUS)
    in_cup2, total = count_particles_in_cup(cup2, liquid, s.CUP2_HEIGHT, s.CUP2_RADIUS)

    print(f"\n=== Final Result ===")
    print(f"Total particles in cup: {in_cup}/{total} ({100*in_cup/total:.1f}%)")
    print(f"Total particles in cup2: {in_cup2}/{total} ({100*in_cup2/total:.1f}%)")
    print(f"Percentage spilled: {100*(1 - (in_cup + in_cup2) / total):.1f}%")

if __name__ == "__main__":
    main()