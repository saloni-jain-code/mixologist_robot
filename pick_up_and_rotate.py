'''
In terminal:
python -m venv robotics
source robotics/bin/activate
pip install -r requirements.txt

python pick_up_and_rotate.py
'''
import os
import re
import json
from dotenv import load_dotenv
import numpy as np
import genesis as gs
import torch
import sys
import matplotlib.pyplot as plt
import settings as s
import random
from helper import (count_particles_in_cup, add_coordinate_frame, pour_drink)
import google.generativeai as genai

load_dotenv()  

BARTENDER_PROMPT = """
You are a robotic bartender.

You ONLY have access to these ingredients:
- Alcohols: Vodka
- Mixers: Tonic, Seltzer, Orange Juice

Each ingredient is in a colored cup.
Vodka is Blue, Tonic is Pink, Seltzer is Green, Orange Juice is Red

You can pour only these amounts:
- "low"
- "medium"
- "high"

Given a user drink request, choose a reasonable combination of the available ingredients
and assign a pour level ("low", "medium", or "high") to each ingredient you use.

OUTPUT FORMAT (IMPORTANT):
- Return ONLY a dictionary object (no prose, no explanations).
- Give the colored cup name, not the ingredient
- Keys are ingredient names as strings.
- Values are one of "low", "medium", "high".
- Do not include any ingredients that are not in the available list.
- Do not wrap code with ```python or ```json.

Example:

User request: "Make me a Moscow mule."
Your Python Dictionary output:
{{
  "blue": "medium",
  "green": "high"
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
    dictionary_response = json.loads(response.text)
    print(dictionary_response)

    pour_level = sys.argv[1] if len(sys.argv) > 1 else "medium"

    print("Pour level (high, medium, or low): ", pour_level)

    pour_level = sys.argv[1] if len(sys.argv) > 1 else "medium"
    print("Pour level (high, medium, or low): ", pour_level)

    ########################## init and create a scene ##########################
    gs.init(
        backend=gs.cpu, 
        logging_level='warning'
    )


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
    # --------------------------------------------------
    # Shelf parameters
    # --------------------------------------------------
    shelf_width  = 0.9   # x size (left-right)
    shelf_depth  = 0.1   # y size (front-back)
    shelf_thick  = 0.02  # z thickness
    leg_radius   = 0.015
    leg_height   = 0.6   # total height of all three layers
    num_layers   = 3

    # Position of the shelf center in world coordinates
    shelf_center = np.array([0.35, -0.6, leg_height / 2.0])

    # y and x half-extent
    hx = shelf_width / 2.0
    hy = shelf_depth / 2.0

    # z positions of the three shelves
    z_bottom = 0.1                      # bottom shelf height (center)
    z_step   = (leg_height - 0.1) / 2.0 # spacing between shelves
    z_layers = [z_bottom + i * z_step for i in range(num_layers)]
    SHELF_LAYER_INDEX = 1  # e.g., middle shelf
    shelf_z_center    = z_layers[SHELF_LAYER_INDEX]
    shelf_top_z       = shelf_z_center + shelf_thick / 2.0

    # --------------------------------------------------
    # Vertical posts (cylinders)
    # --------------------------------------------------
    # corner offsets in x-y
    corner_offsets = [
        np.array([ hx,  hy]),
        np.array([-hx,  hy]),
        np.array([-hx, -hy]),
        np.array([ hx, -hy]),
    ]

    for xy in corner_offsets:
        x = shelf_center[0] + xy[0]
        y = shelf_center[1] + xy[1]
        # cylinder pos is at its center; we want it to touch ground at z=0
        z = leg_height / 2.0
        scene.add_entity(
            gs.morphs.Cylinder(
                pos=(x, y, z),
                radius=leg_radius,
                height=leg_height,
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.4, 0.3, 0.2)),  # wood-ish
        )

    # --------------------------------------------------
    # Shelves (boxes)
    # --------------------------------------------------
    for z in z_layers:
        scene.add_entity(
            gs.morphs.Box(
                pos=(shelf_center[0], shelf_center[1], z),
                size=(shelf_width, shelf_depth, shelf_thick),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7)),
        )

    # rod = scene.add_entity(
    #         morph=gs.morphs.Cylinder(
    #             height=s.ROD_HEIGHT,
    #             radius=s.ROD_RADIUS,
    #             pos=s.ROD_START_POS,  # same height as the cups, but displaced to the right
    #         )
    # )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface = gs.surfaces.Plastic(
            color = (0, 0, 0)
        )                 
    )

    target_cup = scene.add_entity(
        gs.morphs.Mesh(file=s.WHITE_CUP_FILE, pos=s.TARGET_CUP_START_POS, scale=s.TARGET_CUP_SCALE, euler=(90, 0, 0)),
    )

    # Helper: put a cup on the chosen shelf, keeping original x/y
    def cup_on_shelf(x, y, cup_height):
        return np.array([x, y, shelf_top_z + cup_height / 2.0])

    def liquid_on_shelf(x, y, liquid_height):
        return np.array([x, y, shelf_top_z + liquid_height / 2.0])

    # Use original X/Y from your settings, but override Z so they sit on the shelf
    LEFT_CUP_X, LEFT_CUP_Y, _  = s.MID_LEFT_CUP_START_POS
    RIGHT_CUP_X, RIGHT_CUP_Y, _ = s.MID_RIGHT_CUP_START_POS

    MID_LEFT_CUP_SHELF_POS  = cup_on_shelf(LEFT_CUP_X,  LEFT_CUP_Y,  s.CUP_HEIGHT)
    MID_RIGHT_CUP_SHELF_POS = cup_on_shelf(RIGHT_CUP_X, RIGHT_CUP_Y, s.CUP_HEIGHT)
    LOW_LEFT_CUP_SHELF_POS  = cup_on_shelf(LEFT_CUP_X,  LEFT_CUP_Y,  s.CUP_HEIGHT - 0.4)
    LOW_RIGHT_CUP_SHELF_POS = cup_on_shelf(RIGHT_CUP_X, RIGHT_CUP_Y, s.CUP_HEIGHT - 0.4)

    COLOR_CUP_FILES      = [s.BLUE_CUP_FILE, s.RED_CUP_FILE, s.GREEN_CUP_FILE, s.PINK_CUP_FILE]
    COLOR_CUP_POSITIONS  = [MID_LEFT_CUP_SHELF_POS, MID_RIGHT_CUP_SHELF_POS, LOW_LEFT_CUP_SHELF_POS, LOW_RIGHT_CUP_SHELF_POS]
    LIQUID_COLORS        = [s.BLUE, s.RED, s.GREEN, s.PINK]
    RAND                 = 0 # random.randint(0, 1)

    # RED CUP (randomly left or right)
    red_cup_pos = COLOR_CUP_POSITIONS[0]
    red_cup = scene.add_entity(
        gs.morphs.Mesh(
            file=s.RED_CUP_FILE,
            pos=red_cup_pos,
            scale=s.CUP_SCALE,
            euler=(90, 0, 0),
        ),
    )

    red_liquid_pos = liquid_on_shelf(red_cup_pos[0], red_cup_pos[1], s.LIQUID_HEIGHT)
    red_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,
            radius=s.LIQUID_RADIUS,
            pos=red_liquid_pos,
        ),
        surface=gs.surfaces.Default(
            color    = s.RED,
            vis_mode = 'particle',
        ),
    )

    # BLUE CUP (the other shelf position)
    blue_cup_pos = COLOR_CUP_POSITIONS[1] # changed to hardcoded for now
    blue_cup = scene.add_entity(
        gs.morphs.Mesh(
            file=s.BLUE_CUP_FILE,
            pos=blue_cup_pos,
            scale=s.CUP_SCALE,
            euler=(90, 0, 0),
        ),
    )

    blue_liquid_pos = liquid_on_shelf(blue_cup_pos[0], blue_cup_pos[1], s.LIQUID_HEIGHT)
    blue_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,
            radius=s.LIQUID_RADIUS,
            pos=blue_liquid_pos,
        ),
        surface=gs.surfaces.Default(
            color    = s.BLUE,
            vis_mode = 'particle',
        ),
    )

    # GREEN CUP
    green_cup_pos = COLOR_CUP_POSITIONS[2]
    green_cup = scene.add_entity(
        gs.morphs.Mesh(
            file=s.GREEN_CUP_FILE,
            pos=green_cup_pos,
            scale=s.CUP_SCALE,
            euler=(90, 0, 0),
        ),
    )

    green_liquid_pos = liquid_on_shelf(green_cup_pos[0], green_cup_pos[1], s.LIQUID_HEIGHT)
    green_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,
            radius=s.LIQUID_RADIUS,
            pos=green_liquid_pos,
        ),
        surface=gs.surfaces.Default(
            color    = s.GREEN,
            vis_mode = 'particle',
        ),
    )

    # PINK CUP
    pink_cup_pos = COLOR_CUP_POSITIONS[3]
    pink_cup = scene.add_entity(
        gs.morphs.Mesh(
            file=s.PINK_CUP_FILE,
            pos=pink_cup_pos,
            scale=s.CUP_SCALE,
            euler=(90, 0, 0),
        ),
    )

    pink_liquid_pos = liquid_on_shelf(pink_cup_pos[0], pink_cup_pos[1], s.LIQUID_HEIGHT)
    pink_liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=s.LIQUID_HEIGHT,
            radius=s.LIQUID_RADIUS,
            pos=green_liquid_pos,
        ),
        surface=gs.surfaces.Default(
            color    = s.PINK,
            vis_mode = 'particle',
        ),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    )

    cam = scene.add_camera(
        model='pinhole',
        res=(320, 320),
        pos=s.CAM_POS,                         # put camera above robot
        lookat=np.array(s.TARGET_CUP_START_POS)-np.array([0, 0, 0.18]),         # look at the cup # got rid of 
        up=(0,0,1),
        fov=60,
        GUI=False,                             # if True: opens a window with the camera view
        near=0.05,
        far=5.0,
    )

    lookat_point = np.array([0.3, -0.5, 0.5])  # Point between your cups

    # T_w2c = cam.extrinsics
    # print("World to Camera ", T_w2c)
    # T_c2w = np.linalg.inv(T_w2c)
    # print("Camera to World ", T_c2w)
    # extrinsic_matrix = look_at_transform(
    #     pos=np.array(s.CAM_POS),
    #     lookat=np.array(s.CUP_START_POS),
    #     up=np.array([0,-1,0])
    # ) 
    # extrinsic_matrix = np.array(
    #     [[ 1.,  0.,  0.,  0.55], 
    #      [ 0.,  0., -1.,  -0.5],
    #      [ 0.,  1.,  0.,    0.],
    #      [ 0.,  0.,  0.,    1.]]
    # )

    # add_coordinate_frame(scene)
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
    cup_entity_dict = {
        "red": red_cup,
        "blue": blue_cup,
        "green": green_cup,
        "pink": pink_cup,
    }
    for index, (mixer, lvl) in enumerate(dictionary_response.items()):
        next_mixer = None
        if index < len(dictionary_response) - 1:
            next_mixer = list(dictionary_response.items())[index+1][0]
        pour_drink(scene, franka, mixer, cup_entity_dict[mixer], cam, lvl, index, next_mixer)

    # stir liquids in target cup
    # approach(scene, franka, s.ROD_START_POS)
    # grasp(scene, franka)
    # lift(scene, franka, s.ROD_START_POS, s.LIFT_HEIGHT)
    # move_dist(scene, franka, 0, s.X_OFFSET[pour_level])
    # stir(scene, franka)

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