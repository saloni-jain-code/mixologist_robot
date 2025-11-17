'''
In terminal:
python -m venv robotics
source robotics/bin/activate
pip install torch torchvision 
pip install genesis-world

python pick_up_and_rotate.py
'''
import numpy as np
import genesis as gs
import torch
import sys

pour_level = sys.argv[1] if len(sys.argv) > 1 else "medium"
print("Pour level (high, medium, or low): ", pour_level)
pour_levels = {"high":1.4, "medium": 1.625, "low": 1.675}

CUP_START_POS = (0.65, 0.0, 0.12)
CUP2_START_POS = (0.76, 0.0, 0.12)
CUP_SCALE = 0.025
CUP2_SCALE = 0.032
LIQUID_RADIUS = 0.025
LIQUID_HEIGHT = 0.1
LIQUID_START_POS = (CUP_START_POS[0], CUP_START_POS[1], CUP_START_POS[2] + 0.3)
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
        material=gs.materials.PBD.Liquid(
            sampler="regular",
            rho=1.0,
            # density_relaxation=1.0,
            # viscosity_relaxation=0.0,
        ),
        morph=gs.morphs.Cylinder(
            height=LIQUID_HEIGHT,        # 12 cm tall
            radius=LIQUID_RADIUS,        # 3 cm radius
            pos=LIQUID_START_POS,  # sitting on plane (z = height/2)),
        ),
)

cam = scene.add_camera(
    model='pinhole',
    res=(320, 320),
    pos=(0.6, -0.3, 0.9),        # put camera above robot
    lookat=CUP_START_POS,        # look at the cup
    up=(0,0,1),
    fov=60,
    GUI=False,                  # if True: opens a window with the camera view
    near=0.05,
    far=5.0,
)

plane = scene.add_entity(gs.morphs.Plane())

# cup = scene.add_entity(
#     gs.morphs.Cylinder(
#         height=0.12,        # 12 cm tall
#         radius=0.03,        # 3 cm radius
#         pos=CUP_START_POS,  # sitting on plane (z = height/2)
#     )
# )

# cup = scene.add_entity(
#     gs.morphs.MJCF(file='object_sim/cup/object.xml', pos=CUP_START_POS, scale=2),
# )

cup = scene.add_entity(
    gs.morphs.Mesh(file='cup.obj', pos=CUP_START_POS, scale=CUP_SCALE, euler=(90, 0, 0)),
)

cup2 = scene.add_entity(
    gs.morphs.Mesh(file='cup.obj', pos=CUP2_START_POS, scale=CUP2_SCALE, euler=(90, 0, 0)),
)

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

########################## side grasp motion ##########################
end_effector = franka.get_link('hand')

target_pos = np.array((0.65, 0.00, 0.06)) + np.array([0.0, 0.0, 0.03])
approach_dir = np.array([0.0, 1.0, 0.0])  # approach from -Y toward +Y
side_quat = side_grasp_quat(approach_dir, np.array([0,0,1]), order="wxyz")

pregrasp_offset  = -0.14
gripper_offset = np.array([0.0, 0.10, 0.0]) # offset from center of end-effector to center of grip
retreat_distance = 0 # 0.16
open_width  = 0.06
close_force = -1.0

pregrasp_pos = target_pos - approach_dir * pregrasp_offset
grasp_pos    = target_pos.copy() + gripper_offset

# ------------- move to pregrasp ----------------
q_pre = franka.inverse_kinematics(
    link=end_effector, 
    pos=pregrasp_pos, 
    quat=side_quat)
q_pre[-2:] = open_width
path = franka.plan_path(qpos_goal=q_pre, num_waypoints=200)
for wp in path:
    franka.control_dofs_position(wp)
    scene.step()
    
for _ in range(80): scene.step()

# ------------- approach ----------------
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
franka.control_dofs_force(np.array([close_force, close_force]), fingers_dof)
for _ in range(140): scene.step()

# ------------- lift ----------------
lift_height = 0.28
q_lift = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([grasp_pos[0], grasp_pos[1], lift_height]),
    quat=side_quat,
)
franka.control_dofs_position(q_lift[:-2], motors_dof)

for _ in range(200):
    scene.step()
    

# Rotate
joint7_idx = 6   # zero-based index
for i in range(100):
    if i == 0:
        franka.control_dofs_position(
            np.array([pour_levels[pour_level]]),          # target angle in radians
            np.array([joint7_idx]),   # which joint to command
        )
    print("control force:", franka.get_dofs_control_force([joint7_idx]))
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

for _ in range(200):
    scene.step()