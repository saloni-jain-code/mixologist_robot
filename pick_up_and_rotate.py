'''
In terminal:
python -m venv robotics
source robotics/bin/activate
pip install -r requirements.txt

python pick_up_and_rotate.py
'''
import os
from dotenv import load_dotenv
import numpy as np
import genesis as gs
import torch
import sys
import matplotlib.pyplot as plt
import settings as s
import random
from helper import (count_particles_in_cup, approach, grasp, ungrasp, lift, move_dist, rotate, stir, pixel_to_world, get_cup_centers, get_camera_render, get_cup_world_coordinates, pour_drink)
import google.generativeai as genai

load_dotenv()  

BARTENDER_PROMPT = """
You are a robotic bartender.

You ONLY have access to these ingredients:
- Alcohols: Vodka, Gin, Rum
- Mixers: Tonic, Seltzer, Cola

You can pour only these amounts:
- "low"
- "medium"
- "high"

Given a user drink request, choose a reasonable combination of the available ingredients
and assign a pour level ("low", "medium", or "high") to each ingredient you use.

OUTPUT FORMAT (IMPORTANT):
- Return ONLY a JSON object (no prose, no explanations).
- Keys are ingredient names as strings.
- Values are one of "low", "medium", "high".
- Do not include any ingredients that are not in the available list.

Example:

User request: "Make me a Moscow mule."
Your JSON output:
{{
  "Vodka": "medium",
  "Seltzer": "high"
}}

Now respond for this user request:
"{request}"
"""

def main(): 
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel("gemini-2.5-flash")
    user_text = ""

    user_text = input("User: ")

    prompt = BARTENDER_PROMPT.format(request=user_text)
    response = model.generate_content(prompt)

    print(response.text)

    pour_level = sys.argv[1] if len(sys.argv) > 1 else "medium"

    print("Pour level (high, medium, or low): ", pour_level)

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
    COLOR_CUP_POSITIONS = [s.LEFT_CUP_START_POS, s.RIGHT_CUP_START_POS]
    LIQUID_COLORS = [s.BLUE, s.RED]
    RAND = random.randint(0,1)

    red_cup = scene.add_entity(
        gs.morphs.Mesh(file=s.RED_CUP_FILE, pos=COLOR_CUP_POSITIONS[RAND], scale=s.CUP_SCALE, euler=(90, 0, 0)),
    )

    red_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,        # 12 cm tall
            radius=s.LIQUID_RADIUS,        # 3 cm radius
            pos=COLOR_CUP_POSITIONS[RAND] + np.array([0.,0., 0.3]),  # sitting on plane (z = height/2)),
        ),
        surface = gs.surfaces.Default(
            color    = s.RED,
            vis_mode = 'particle'
        )
    )

    blue_cup = scene.add_entity(
        gs.morphs.Mesh(file=s.BLUE_CUP_FILE, pos=COLOR_CUP_POSITIONS[RAND ^ 1], scale=s.CUP_SCALE, euler=(90, 0, 0)),
    )

    blue_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,        # 12 cm tall
            radius=s.LIQUID_RADIUS,        # 3 cm radius
            pos=COLOR_CUP_POSITIONS[RAND ^ 1] + np.array([0.,0., 0.3]),  # sitting on plane (z = height/2)),
        ),
        surface = gs.surfaces.Default(
            color    = s.BLUE,
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

    # cups, depth = get_camera_render(cam)

    # blue_cup_x = cups["blue"][0][0]
    # blue_cup_y = cups["blue"][0][1]

    # red_cup_x = cups["red"][0][0]
    # red_cup_y = cups["red"][0][1]

    # # detect if red cup or blue cup is the left cup
    # if blue_cup_x < red_cup_x:
    #     left_cup_x = blue_cup_x
    #     left_cup_y = blue_cup_y
    #     right_cup_x = red_cup_x
    #     right_cup_y = red_cup_y
    # else: 
    #     left_cup_x = red_cup_x
    #     left_cup_y = red_cup_y
    #     right_cup_x = blue_cup_x
    #     right_cup_y = blue_cup_y
    
    # pour left cup first
    ratios = {
        "red": "high",
        "blue": "low"
    }
    cup_entity_dict = {
        "red": red_cup,
        "blue": blue_cup
    }
    for index, (mixer, lvl) in enumerate(ratios.items()):
        next_mixer = None
        if index < len(ratios) - 1:
            next_mixer = list(ratios.items())[index+1][0]
        pour_drink(scene, franka, mixer, cup_entity_dict[mixer], cam, lvl, next_mixer)
    
    # left_cup_pos = pixel_to_world(K, left_cup_x, left_cup_y)
    # approach(scene, franka, left_cup_pos)
    # grasp(scene, franka)
    # lift(scene, franka, left_cup.get_pos().cpu().numpy(), s.LIFT_HEIGHT)
    # move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    # rotate(scene, franka, pour_level, 1)
    # move_dist(scene, franka, 0, -1.0 * s.X_OFFSET[pour_level])
    # lift(scene, franka, left_cup.get_pos().cpu().numpy(), 0.08)
    # ungrasp(scene, franka)
    # move_dist(scene, franka, 1, 0.06)
    # move_dist(scene, franka, 0, 0.3, 100)

    # pour right cup next
    # right_cup_pos = pixel_to_world(K, right_cup_x, right_cup_y)
    # approach(scene, franka, right_cup_pos)
    # grasp(scene, franka)
    # lift(scene, franka, right_cup_pos, s.LIFT_HEIGHT)
    # move_dist(scene, franka, 0, -s.X_OFFSET[pour_level]) # x_offset should depend on pour level + what cup it is 
    # rotate(scene, franka, pour_level, -1)
    # move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    # lift(scene, franka, right_cup.get_pos().cpu().numpy(), 0.08)
    # ungrasp(scene, franka)
    # move_dist(scene, franka, 1, 0.06)

    # stir liquids in target cup
    # approach(scene, franka, s.ROD_START_POS)
    # grasp(scene, franka)
    # lift(scene, franka, s.ROD_START_POS, s.LIFT_HEIGHT)
    # move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    # stir(scene, franka)

    # print("----------")
    # print("PREDICTED LEFT CUP POS: ", left_cup_pos)
    # print("ACTUAL LEFTCUP POS: ", s.LEFT_CUP_START_POS)
    # print("----------")
    # print("PREDICTED RIGHT CUP POS: ", right_cup_pos)
    # print("ACTUAL RIGHT CUP POS: ", s.RIGHT_CUP_START_POS)
    # print("----------\n")

    in_red_cup, total_red = count_particles_in_cup(red_cup, red_liquid, s.CUP_HEIGHT, s.CUP_RADIUS)
    in_blue_cup, total_blue = count_particles_in_cup(blue_cup, blue_liquid, s.CUP_HEIGHT, s.CUP_RADIUS)
    red_in_target_cup, total = count_particles_in_cup(target_cup, red_liquid, s.TARGET_CUP_HEIGHT, s.TARGET_CUP_RADIUS)
    blue_in_target_cup, total = count_particles_in_cup(target_cup, blue_liquid, s.TARGET_CUP_HEIGHT, s.TARGET_CUP_RADIUS)
    total_in_target_cup = red_in_target_cup + blue_in_target_cup
    total_particles = total_red + total_blue
    # print(f"\n=== Final Particle Results ===")
    print(f"Total particles in red cup: {in_red_cup}/{total_red} ({100*in_red_cup/total_red:.1f}%)")
    print(f"Total particles in blue cup: {in_blue_cup}/{total_blue} ({100*in_blue_cup/total_blue:.1f}%)")
    print(f"Total particles in target cup: {total_in_target_cup}/{total_particles} ({100*total_in_target_cup/total_particles:.1f}%)")
    print(f"Percentage spilled: {100*(1 - (in_blue_cup + in_red_cup + total_in_target_cup) / total_particles):.1f}%")

if __name__ == "__main__":
    main()