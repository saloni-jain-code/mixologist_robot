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

pour_level = sys.argv[1] if len(sys.argv) > 1 else "medium"
print("Pour level (high, medium, or low): ", pour_level)
pour_levels = {"high":1, "medium": 1.09, "low": 1.105}
pour_speed = {"high":300, "medium": 225, "low": 200}
x_offset = {"high": 0.02, "medium": 0.035, "low": 0.04} 

CUP_FILE = 'cup.obj'

# Cup bounds (approximate cylinder bounds)
CUP_HEIGHT = 0.1
CUP_RADIUS = 0.03

CUP_START_POS = (0.65, 0.0, 0.12)
CUP_SCALE = 0.025

# Cup2 bounds (approximate cylinder bounds)
CUP2_HEIGHT = 0.12 
CUP2_RADIUS = 0.03

CUP2_START_POS = (0.76, 0.0, 0.12)
CUP2_SCALE = 0.028

CUP3_START_POS = (0.55, 0.0, 0.12)

CUP_TEST_START_POS = (0.9, 0.0, 0.12)

LIQUID_RADIUS = 0.02
LIQUID_HEIGHT = 0.2
LIQUID1_START_POS = (CUP_START_POS[0], CUP_START_POS[1], CUP_START_POS[2] + 0.3)
LIQUID2_START_POS = (CUP3_START_POS[0], CUP3_START_POS[1], CUP3_START_POS[2] + 0.3)

CAM_POS = (0, 0, 0)

lift_height = 0.28

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
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

########################## entities ##########################
liquid = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=LIQUID_HEIGHT,        # 12 cm tall
            radius=LIQUID_RADIUS,        # 3 cm radius
            pos=LIQUID1_START_POS,  # sitting on plane (z = height/2)),
        ),
        surface = gs.surfaces.Default(
            color    = (1.0, 0.4, 0.4),
            vis_mode = 'particle'
        )
)

liquid2 = scene.add_entity(
        material=gs.materials.PBD.Liquid(),
        morph=gs.morphs.Cylinder(
            height=LIQUID_HEIGHT,        # 12 cm tall
            radius=LIQUID_RADIUS,        # 3 cm radius
            pos=LIQUID2_START_POS,  # sitting on plane (z = height/2)),
        ),
        surface = gs.surfaces.Default(
            color    = (0.3, 0.3, 1.0),
            vis_mode = 'particle'
        )
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

plane = scene.add_entity(gs.morphs.Plane())


cup = scene.add_entity(
    gs.morphs.Mesh(file=CUP_FILE, pos=CUP_START_POS, scale=CUP_SCALE, euler=(90, 0, 0)),
)

cup2 = scene.add_entity(
    gs.morphs.Mesh(file=CUP_FILE, pos=CUP2_START_POS, scale=CUP2_SCALE, euler=(90, 0, 0)),
)

cup3 = scene.add_entity(
    gs.morphs.Mesh(file=CUP_FILE, pos=CUP3_START_POS, scale=CUP_SCALE, euler=(90, 0, 0)),
)

# cupTest = scene.add_entity(
#     gs.morphs.Cylinder(
#         height=CUP2_HEIGHT,
#         radius=CUP2_RADIUS,
#         pos=CUP_TEST_START_POS,
#     )
# )

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)



########################## build ##########################
scene.build()

motors_dof  = np.arange(7)
fingers_dof = np.arange(7, 9)

# --- control gains ----------------------------------------------------------
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)

########################## helpers ##########################

def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def mat_to_quat_wxyz(R):
    """Rotation matrix → quaternion [w,x,y,z]."""
    m00,m01,m02 = R[0,0], R[0,1], R[0,2]
    m10,m11,m12 = R[1,0], R[1,1], R[1,2]
    m20,m21,m22 = R[2,0], R[2,1], R[2,2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return np.array([w,x,y,z])

def side_grasp_quat(approach_dir_world, up_hint_world=np.array([0,0,1]), order="wxyz"):
    """
    Build an orientation so hand -Z points along approach_dir_world.
    'up_hint_world' controls roll (jaws vertical if up=[0,0,1]).
    """
    z_hand = -unit(approach_dir_world)
    x_hand = unit(up_hint_world - np.dot(up_hint_world, z_hand) * z_hand)
    y_hand = np.cross(z_hand, x_hand)
    R = np.column_stack([x_hand, y_hand, z_hand])
    q_wxyz = mat_to_quat_wxyz(R)
    if order == "wxyz":
        return q_wxyz
    else:
        w,x,y,z = q_wxyz
        return np.array([x,y,z,w])

def count_particles_in_cup(cup, cup_height, cup_radius):
    """
    Count how many liquid particles are inside cup2's cylindrical bounds.
    """
    # Get particle positions (shape: [n_particles, 3])
    particle_pos = liquid.get_particles_pos().cpu().numpy()
    print("PARTICLE POS SHAPE:", particle_pos.shape)
    print("PARTICLE POS:", particle_pos)
    writer = open("particle_positions.txt", "w")
    for p in particle_pos:
        writer.write(f"{p[0]}, {p[1]}, {p[2]}\n")
    writer.close()
    # Get cup2's current position
    cup_pos = cup.get_pos().cpu().numpy()
    
    # Calculate relative positions (x, y, z) from cup2 center
    rel_pos = particle_pos - cup_pos
    
    # Check cylinder bounds:
    # 1. Radial distance from cup2's center (x-y plane)
    radial_dist = np.sqrt(rel_pos[:, 0]**2 + rel_pos[:, 1]**2)
    
    # 2. Height within cup (z axis) - cup opening at top
    # Assuming cup is upright with opening at +z direction
    z_min = -cup_height / 2  # bottom of cup
    z_max = cup_height / 2   # top of cup (opening)
    
    # Particles are inside if:
    # - radial distance < radius
    # - z is between bottom and top
    inside_radially = radial_dist < cup_radius
    inside_vertically = (rel_pos[:, 2] > z_min) & (rel_pos[:, 2] < z_max)
    
    particles_inside = np.sum(inside_radially & inside_vertically)
    
    return particles_inside, len(particle_pos)


########################## side grasp motion ##########################
end_effector = franka.get_link('hand')
# CUP POS: (0.65, 0.0, 0.12)
target_pos = np.array(CUP_START_POS) - np.array([0.0, 0.0, 0.03])

# target_pos = np.array((0.65, 0.00, 0.06)) + np.array([0.0, 0.0, 0.03])
approach_dir = np.array([0.0, 1.0, 0.0])  # approach from -Y toward +Y
side_quat = side_grasp_quat(approach_dir, np.array([0,0,1]), order="wxyz")

pregrasp_offset  = -0.14
gripper_offset = np.array([0.0, 0.10, 0.0]) # offset from center of end-effector to center of grip
retreat_distance = 0 # 0.16
open_width  = 0.06
close_force = -0.2

pregrasp_pos = target_pos - approach_dir * pregrasp_offset
grasp_pos    = target_pos.copy() + gripper_offset

# ------------- get camera images ----------------
# rgb_arr, depth_arr, seg_arr, normal_arr = cam.render()
# plt.imshow(rgb_arr)
# plt.axis('off')
# plt.show()


def approach(franka, cup_pos):
    # -- preapproach
    target_pos = np.array(cup_pos) - np.array([0.0, 0.0, 0.03])
    pregrasp_pos = target_pos - approach_dir * pregrasp_offset
    end_effector = franka.get_link('hand')
    q_pre = franka.inverse_kinematics(
            link=end_effector, 
            pos=pregrasp_pos, 
            quat=side_quat
            )
    q_pre[-2:] = open_width
    path = franka.plan_path(qpos_goal=q_pre, num_waypoints=200)
    for wp in path:
        franka.control_dofs_position(wp)
        scene.step()

    # ------------- approach ----------------
    for _ in range(80): scene.step()
    n_cart_steps_in = 20
    for i in range(1, n_cart_steps_in + 1):
        a = i / n_cart_steps_in
        p = (1 - a) * pregrasp_pos + a * grasp_pos # straight line interpolation
        q = franka.inverse_kinematics(link=end_effector, pos=p, quat=side_quat)
        q[-2:] = open_width # last two joints are gripper positions, so this just keeps gripper open
        franka.control_dofs_position(q)
        scene.step()
    for _ in range(20): scene.step()



# ------------- grasp ----------------
def grasp(franka):
    franka.control_dofs_force(np.array([close_force, close_force]), fingers_dof)
    for _ in range(70): scene.step() # before it was 140

def ungrasp(franka):
    n_steps = 40
    
    for i in range(n_steps):
        if i == 0:
            franka.control_dofs_position([0.1, 0.1], fingers_dof)
        if i == (n_steps // 2):
            franka.control_dofs_position([0.2, 0.2], fingers_dof)
        scene.step()
    for _ in range(70): scene.step() # before it was 140

# ------------- lift ----------------
def lift(franka, cup_pos, lift_height):
    end_effector = franka.get_link('hand')
    
    target_pos = np.array(cup_pos) - np.array([0.0, 0.0, 0.03])
    current_pos = end_effector.get_pos().cpu().numpy()
    grasp_pos    = target_pos.copy() + gripper_offset
    new_grasp_pos = np.array([grasp_pos[0], grasp_pos[1], lift_height])
    n_move_steps = 100
    for i in range(n_move_steps):
        alpha = i / n_move_steps
        intermediate_pos = (1 - alpha) * current_pos + alpha * new_grasp_pos
        q_lift = franka.inverse_kinematics(
            link=end_effector,
            pos=intermediate_pos,
            quat=side_quat,
        )
        franka.control_dofs_position(q_lift[:-2], motors_dof)
        scene.step()
    for _ in range(100):
        scene.step()

def move_dist(franka, direction, dist):
    '''
    direction = 0, 1, 2 to represent x, y, z respectively

    dist is the distance 
    '''
    end_effector = franka.get_link('hand')
    current_pos = end_effector.get_pos().cpu().numpy()
    target_pos_closer = current_pos.copy()
    target_pos_closer[direction] += dist  # move dist in +direction
    n_move_steps = 50

    for i in range(n_move_steps):
        alpha = i / n_move_steps
        intermediate_pos = (1 - alpha) * current_pos + alpha * target_pos_closer
        
        q_move = franka.inverse_kinematics(
            link=end_effector,
            pos=intermediate_pos,
            quat=side_quat,
        )
        
        franka.control_dofs_position(q_move[:-2], motors_dof)
        scene.step()
    
    for _ in range(30):
        scene.step()


# ------------- rotate to pour ----------------
def rotate(franka, pour_level):
    joint7_idx = 6
    current_angle = franka.get_dofs_position([joint7_idx]).cpu().numpy()[0]
    target_angle = pour_levels[pour_level]
    n_steps = pour_speed[pour_level]  # More steps = slower rotation

    for i in range(n_steps):
        # Smoothly interpolate from current to target angle
        alpha = i / n_steps
        intermediate_angle = (1 - alpha) * current_angle + alpha * target_angle
        
        franka.control_dofs_position(
            np.array([intermediate_angle]),
            np.array([joint7_idx]),
        )
        scene.step()

    for _ in range(50):
        scene.step()

    for i in range(100):
        if i == 0:
            franka.control_dofs_position(
                np.array([2.7]),          # target angle in radians
                np.array([joint7_idx]),   # which joint to command
            )
        print("control force:", franka.get_dofs_control_force([joint7_idx]))
        scene.step()

    for _ in range(40):
        scene.step()

approach(franka, CUP_START_POS)
grasp(franka)
lift(franka, CUP_START_POS, lift_height)
move_dist(franka, 0, x_offset[pour_level])
rotate(franka, pour_level)
move_dist(franka, 0, -x_offset[pour_level])
lift(franka, cup.get_pos().cpu().numpy(), 0.08)
ungrasp(franka)
move_dist(franka, 1, 0.05)

# approach(franka, CUP3_START_POS)
# grasp(franka)
# lift(franka, CUP3_START_POS, lift_height)
# move_horizontally(franka, x_offset[pour_level]) # x_offset should depend on pour level + what cup it is 
# rotate(franka, pour_level)
# move_horizontally(franka, -x_offset[pour_level])
# lift(franka, cup3.get_pos().cpu().numpy(), 0.08)

in_cup, total = count_particles_in_cup(cup, CUP_HEIGHT, CUP_RADIUS)
in_cup2, total = count_particles_in_cup(cup2, CUP2_HEIGHT, CUP2_RADIUS)

print(f"\n=== Final Result ===")
print(f"Total particles in cup: {in_cup}/{total} ({100*in_cup/total:.1f}%)")
print(f"Total particles in cup2: {in_cup2}/{total} ({100*in_cup2/total:.1f}%)")
print(f"Percentage spilled: {100*(1 - (in_cup + in_cup2) / total):.1f}%")