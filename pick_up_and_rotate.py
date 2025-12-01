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
import random
from helper import (count_particles_in_cup, approach, grasp, ungrasp, lift, move_dist, rotate, stir, pixel_to_world, get_cup_centers, get_camera_render, get_cup_world_coordinates)

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
    rod = scene.add_entity(
            morph=gs.morphs.Cylinder(
                height=s.ROD_HEIGHT,
                radius=s.ROD_RADIUS,
                pos=s.ROD_START_POS,  # same height as the cups, but displaced to the right
            )
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface = gs.surfaces.Plastic(
            color = (0, 0, 0)
        )                 
    )

    target_cup = scene.add_entity(
        gs.morphs.Mesh(file=s.WHITE_CUP_FILE, pos=s.TARGET_CUP_START_POS, scale=s.TARGET_CUP_SCALE, euler=(90, 0, 0)),
    )

    # LEFT_CUP_START_POS = (0.3, 0.0, 0.0)
    # RIGHT_CUP_START_POS = (0.6, 0.0, 0.0)
    # replace with random positions once get_cup_centers is implemented
    # LEFT_CUP_START_POS = (np.random.uniform(0.4, 0.6), 0.0, 0.0)
    # RIGHT_CUP_START_POS = (np.random.uniform(0.7, 0.9), 0.0, 0.0)
    COLOR_CUP_FILES = [s.BLUE_CUP_FILE, s.RED_CUP_FILE]
    RAND = random.randint(0,1)
    if RAND == 0:
        LEFT_LIQUID_COLOR = (0.3, 0.3, 1.0)
        RIGHT_LIQUID_COLOR = (1.0, 0.4, 0.4)
    else:
        LEFT_LIQUID_COLOR = (1.0, 0.4, 0.4)
        RIGHT_LIQUID_COLOR = (0.3, 0.3, 1.0)

    left_cup = scene.add_entity(
        gs.morphs.Mesh(file=COLOR_CUP_FILES[RAND], pos=s.LEFT_CUP_START_POS, scale=s.CUP_SCALE, euler=(90, 0, 0)),
    )
    left_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,        # 12 cm tall
            radius=s.LIQUID_RADIUS,        # 3 cm radius
            pos=s.LEFT_CUP_START_POS + np.array([0.,0., 0.3]),  # sitting on plane (z = height/2)),
        ),
        surface = gs.surfaces.Default(
            color    = LEFT_LIQUID_COLOR,
            vis_mode = 'particle'
        )
    )

    right_cup = scene.add_entity(
        gs.morphs.Mesh(file=COLOR_CUP_FILES[RAND ^ 1], pos=s.RIGHT_CUP_START_POS, scale=s.CUP_SCALE, euler=(90, 0, 0)),
    )

    right_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,        # 12 cm tall
            radius=s.LIQUID_RADIUS,        # 3 cm radius
            pos=s.RIGHT_CUP_START_POS + np.array([0.,0., 0.3]),  # sitting on plane (z = height/2)),
        ),
        surface = gs.surfaces.Default(
            color    = RIGHT_LIQUID_COLOR,
            vis_mode = 'particle'
        )
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )

    cam = scene.add_camera(
        model='pinhole',
        res=(320, 320),
        pos=s.CAM_POS,                         # put camera above robot
        lookat=s.TARGET_CUP_START_POS,         # look at the cup
        up=(0,0,1),
        fov=60,
        GUI=False,                             # if True: opens a window with the camera view
        near=0.05,
        far=5.0,
    )
    K = cam.intrinsics
    # extrinsic_matrix = look_at_transform(
    #     pos=np.array(s.CAM_POS),
    #     lookat=np.array(s.CUP_START_POS),
    #     up=np.array([0,-1,0])
    # ) 
    extrinsic_matrix = np.array(
        [[ 1.,  0.,  0.,  0.55], 
         [ 0.,  0., -1.,  -0.5],
         [ 0.,  1.,  0.,    0.],
         [ 0.,  0.,  0.,    1.]]
    )

    scene.build()

    # --- control gains ----------------------------------------------------------
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    )

    ########################## EXECUTION PIPELINE ##########################
    for _ in range(50):
        scene.step()

    cups, depth = get_camera_render(cam)

    blue_cup_x = cups["blue"][0][0]
    blue_cup_y = cups["blue"][0][1]

    red_cup_x = cups["red"][0][0]
    red_cup_y = cups["red"][0][1]

    # detect if red cup or blue cup is the left cup
    if blue_cup_x < red_cup_x:
        left_cup_x = blue_cup_x
        left_cup_y = blue_cup_y
        right_cup_x = red_cup_x
        right_cup_y = red_cup_y
    else: 
        left_cup_x = red_cup_x
        left_cup_y = red_cup_y
        right_cup_x = blue_cup_x
        right_cup_y = blue_cup_y
    
    # pour left cup first
    left_cup_pos = pixel_to_world(K, left_cup_x, left_cup_y)
    approach(scene, franka, left_cup_pos)
    grasp(scene, franka)
    lift(scene, franka, left_cup.get_pos().cpu().numpy(), s.LIFT_HEIGHT)
    move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    rotate(scene, franka, pour_level, 1)
    move_dist(scene, franka, 0, -1.0 * s.X_OFFSET[pour_level])
    lift(scene, franka, left_cup.get_pos().cpu().numpy(), 0.08)
    ungrasp(scene, franka)
    move_dist(scene, franka, 1, 0.06)
    move_dist(scene, franka, 0, 0.3, 100)

    # pour right cup next
    right_cup_pos = pixel_to_world(K, right_cup_x, right_cup_y)
    approach(scene, franka, right_cup_pos)
    grasp(scene, franka)
    lift(scene, franka, right_cup_pos, s.LIFT_HEIGHT)
    move_dist(scene, franka, 0, -s.X_OFFSET[pour_level]) # x_offset should depend on pour level + what cup it is 
    rotate(scene, franka, pour_level, -1)
    move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    lift(scene, franka, right_cup.get_pos().cpu().numpy(), 0.08)
    ungrasp(scene, franka)
    move_dist(scene, franka, 1, 0.06)

    # stir liquids in target cup
    # approach(scene, franka, s.ROD_START_POS)
    # grasp(scene, franka)
    # lift(scene, franka, s.ROD_START_POS, s.LIFT_HEIGHT)
    # move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    # stir(scene, franka)

    print("----------")
    print("PREDICTED LEFT CUP POS: ", left_cup_pos)
    print("ACTUAL LEFTCUP POS: ", s.LEFT_CUP_START_POS)
    print("----------")
    print("PREDICTED RIGHT CUP POS: ", right_cup_pos)
    print("ACTUAL RIGHT CUP POS: ", s.RIGHT_CUP_START_POS)
    print("----------\n")

    in_left_cup, total = count_particles_in_cup(left_cup, left_liquid, s.CUP_HEIGHT, s.CUP_RADIUS)
    in_target_cup, total = count_particles_in_cup(target_cup, left_liquid, s.TARGET_CUP_HEIGHT, s.TARGET_CUP_RADIUS)

    print(f"\n=== Final Particle Results ===")
    print(f"Total particles in left cup: {in_left_cup}/{total} ({100*in_left_cup/total:.1f}%)")
    print(f"Total particles in target cup: {in_target_cup}/{total} ({100*in_target_cup/total:.1f}%)")
    print(f"Percentage spilled: {100*(1 - (in_left_cup + in_target_cup) / total):.1f}%")

if __name__ == "__main__":
    main()