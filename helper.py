import numpy as np
import matplotlib.pyplot as plt
import settings
import cv2
from image_detection import detect_colored_cups

motors_dof  = np.arange(7)
fingers_dof = np.arange(7, 9)

target_pos = np.array(settings.TARGET_CUP_START_POS) - np.array([0.0, 0.0, 0.03])
approach_dir = np.array([0.0, 1.0, 0.0])  # approach from -Y toward +Y

#-----------------------------------------------------------------------
#   MOTOR FUNCTIONS
#-----------------------------------------------------------------------
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

def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

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
side_quat = side_grasp_quat(approach_dir, np.array([0,0,1]), order="wxyz")

pregrasp_offset  = -0.4 # -0.14
gripper_offset = np.array([0.0, 0.10, 0.0]) # offset from center of end-effector to center of grip
retreat_distance = 0 # 0.16
open_width  = 0.06
close_force = -0.3 # was -0.2 before

# pregrasp_pos = target_pos - approach_dir * pregrasp_offset
grasp_pos    = target_pos.copy() + gripper_offset

def approach(scene, franka, cup_pos):
    # -- preapproach
    target_pos = np.array(cup_pos) + np.array([0.0, -0.00, -0.05])
    pregrasp_pos = target_pos - approach_dir * pregrasp_offset
    grasp_pos    = target_pos.copy() + gripper_offset
    end_effector = franka.get_link('hand')
    q_pre = franka.inverse_kinematics(
            link=end_effector, 
            pos=pregrasp_pos, 
            quat=side_quat
    )
    q_pre[-2:] = open_width
    path = franka.plan_path(qpos_goal=q_pre, num_waypoints=500)
    print("Pregrasp, ", pregrasp_pos)
    print("Grasp pos, ", grasp_pos)

    for wp in path:
        franka.control_dofs_position(wp)
        scene.step()

    # ------------- approach ----------------
    for _ in range(80): scene.step()
    n_cart_steps_in = 200 #20
    for i in range(1, n_cart_steps_in + 1):
        a = i / n_cart_steps_in
        p = (1 - a) * pregrasp_pos + a * grasp_pos # straight line interpolation
        q = franka.inverse_kinematics(link=end_effector, pos=p, quat=side_quat)
        q[-2:] = open_width # last two joints are gripper positions, so this just keeps gripper open
        franka.control_dofs_position(q)
        scene.step()
    for _ in range(20): scene.step()

'''
media_pipe creates a bouding box around the cup and gives the coordiates of the box
we find the center of the box or the bottom of the box (1x3 image coordinate)
we input the 1x3 image coordinates into this function, which returns 1x3 world coordinates
we input the world coordinates into the approach function
'''
def get_cup_world_coordinates(intrinsic_matrix, extrinsic_matrix, image_x, image_y, depth):
    '''
    cup_image_coordinates: a center of the coordinates of the cup
    return cup world coordinates
    cup_image_coordinates
    '''
    # point_cloud = depth_to_camera_frame_point_cloud(cup_image_coordinates, intrinsic_matrix)
    # return transform_camera_to_world(point_cloud, extrinsic_matrix)
    camera_coordinates = depth_image_to_camera_frame(intrinsic_matrix, image_x, image_y, depth)
    return transform_camera_to_world(extrinsic_matrix, camera_coordinates) 

# ------------- get camera images ----------------
def get_camera_render(cam):
    '''
    returns an array of dictionaries containing the various colors of cups we detect as well as a depth array
    for example {"red":[(cx,cy),...], "blue":[...], ...}
    {'red': [(243, 126)], 'green': [], 'blue': [(76, 126)]}
    '''
    rgb_arr, depth_arr, seg_arr, normal_arr = cam.render(depth=True)
    # print("RGB", rgb_arr[0])
    # print("DEPTH", depth_arr[0])
    # print("DEPTH", depth_arr[1])
    bgr_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    cups = detect_colored_cups(bgr_frame)
    print(cups)
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2GRAY)

    blue_cup_x = cups["blue"][0][0]
    blue_cup_y = cups["blue"][0][1]

    red_cup_x = cups["red"][0][0]
    red_cup_y = cups["red"][0][1]

    # blue_cup_x = 76
    # blue_cup_y = 126

    # red_cup_x = 243
    # red_cup_y = 126

    plt.plot(red_cup_x, red_cup_y, 'ro') 
    plt.plot(blue_cup_x, blue_cup_y, 'bo') 
    plt.imshow(gray)
    plt.show()

    
    return cups, depth_arr
    # print("DEPTH ARR SHAPE", depth_arr.shape)
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2GRAY)
    plt.plot(160, 150, 'ro') 
    plt.imshow(gray)
    plt.show()
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) #try wo inv.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #---------
    # Make a copy of the original image to draw on
    img_contours = rgb_arr.copy()  # or gray.copy() if you want grayscale background

    # Draw all contours in green with thickness 2
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    # If using matplotlib, convert BGR to RGB
    img_contours_rgb = cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)

    # Show the image with contours
    plt.figure(figsize=(8,6))
    plt.imshow(img_contours_rgb)
    plt.axis('off')
    plt.show() 
    #----------

    # cup_centers_px = []
    # for cnt in contours:
    #     M = cv2.moments(cnt)
    #     if M["m00"] > 0:
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])
    #         cup_centers_px.append((cx, cy))
    # print("CUP CENTERS PX:", cup_centers_px)

    # plt.imshow(rgb_arr)
    # plt.axis('off')
    # plt.show()
    # pc, mask_arr = cam.render_pointcloud()
    

'''
INTRINSIC MATRIX 
[[277.12812921   0.         160.        ]
 [  0.         277.12812921 160.        ]
 [  0.           0.           1.        ]]
TRANSFORMATION MATRIX 
[[-1.    0.    0.    0.65]
 [-0.    1.    0.   -0.5 ]
 [-0.    0.    1.    0.  ]
 [ 0.    0.    0.    1.  ]]

'''
def compute_camera_rotation(pos, lookat, up_hint): # generated by Chat, need to validate 
    pos = np.array(pos, float)
    lookat = np.array(lookat, float)
    up_hint = np.array(up_hint, float)

    # +Z forward
    f = lookat - pos
    f = f / np.linalg.norm(f)

    # +X right
    r = np.cross(f, up_hint)
    r = r / np.linalg.norm(r)

    # recompute +Y up (orthogonalized)
    u = np.cross(r, f)

    # Construct rotation: columns are r, u, f
    R_c2w = np.column_stack((r, u, f))
    return R_c2w

def depth_image_to_camera_frame(K, image_x, image_y, depth):
    # depth_arr 320 x 320
    # depth_arr[0] = 1x320
    # depth: scalar
    # K: 3x3 camera intrinsics matrix
    # return: Nx3 point cloud in the camera frame
    ### TODO YOUR CODE STARTS ###

    # # compute x, y in camera frame
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    x = (image_x - cx) * depth / fx
    y = (image_y - cy) * depth / fy
    ### TODO YOUR CODE ENDS ###

    return np.array([x, y, depth])

def transform_camera_to_world(extrinsic_matrix, camera_coordinates):
    # points: Nx3 point cloud in the camera frame
    # extrinsic_matrix: 4x4 camera-to-world extrinsic matrix
    # return: Nx3 point cloud in the world frame

    ### TODO YOUR CODE STARTS ###
    extrinsic_matrix_inv = np.linalg.inv(extrinsic_matrix) # invert
    point_homogenous = np.hstack([camera_coordinates, np.ones(1)]) # homegenize the camera points 1x4 vector
    print("HOMOGENOUS POINT:", point_homogenous)
    world_point_homogeneous = point_homogenous @ extrinsic_matrix_inv  # transform the camera points
    print("WORLD POINT HOMOGENOUS:", world_point_homogeneous)
    pt = world_point_homogeneous[:3] # un-homogenize
    ### TODO YOUR CODE ENDS ###

    return pt

# ------------- grasp ----------------
def grasp(scene, franka):
    franka.control_dofs_force(np.array([close_force, close_force]), fingers_dof)
    for _ in range(70): scene.step() # before it was 140

def ungrasp(scene, franka):
    n_steps = 40
    
    for i in range(n_steps):
        if i == 0:
            franka.control_dofs_position([0.1, 0.1], fingers_dof)
        if i == (n_steps // 2):
            franka.control_dofs_position([0.2, 0.2], fingers_dof)
        scene.step()
    for _ in range(70): scene.step() # before it was 140

# ------------- lift ----------------
def lift(scene, franka, cup_pos, lift_height):
    '''
    lift function lifts franka lift_height relative to where it currently is.
    '''
    end_effector = franka.get_link('hand')
    
    target_pos = np.array(cup_pos) + np.array([0.0, 0.0, 0.09])
    current_pos = end_effector.get_pos().cpu().numpy()
    grasp_pos    = target_pos.copy() + gripper_offset
    new_grasp_pos = np.array([grasp_pos[0], grasp_pos[1], current_pos[2] + lift_height])
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

def move_dist(scene, franka, direction, dist, n_move_steps=50):
    '''
    direction = 0, 1, 2 to represent x, y, z respectively

    dist is the distance 
    '''
    end_effector = franka.get_link('hand')
    current_pos = end_effector.get_pos().cpu().numpy()
    target_pos_closer = current_pos.copy()
    target_pos_closer[direction] += dist  # move dist in +direction
    # n_move_steps = 50

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
def rotate(scene, franka, pour_level, direction=1):
    '''
    direction = 1 for clockwise, -1 for counterclockwise
    pour_level = "high", "medium", "low"
    '''
    joint7_idx = 6
    current_angle = franka.get_dofs_position([joint7_idx]).cpu().numpy()[0]
    # limits = franka.get_dof_limits([joint7_idx])
    # print(f"Joint 7 limits: {limits}")

    # print("CURRENT ANGLE:", current_angle)
    target_angle = direction*settings.POUR_LEVELS[pour_level]
    # the commented out code below doesn't work
    # if direction == -1:
    #     target_angle = current_angle + abs(target_angle)
    # else:
    #     target_angle = current_angle - abs(target_angle)
    # print("TARGET ANGLE:", target_angle)
    n_steps = settings.POUR_SPEED[pour_level]  # More steps = slower rotation

    for i in range(n_steps):
        # Smoothly interpolate from current to target angle
        alpha = i / n_steps
        intermediate_angle = (1 - alpha) * current_angle + alpha * target_angle
        # print("intermediate angle:", intermediate_angle)
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

def count_particles_in_cup(cup, liquid, cup_height, cup_radius):
    """
    Count how many liquid particles are inside cup2's cylindrical bounds.
    """
    # Get particle positions (shape: [n_particles, 3])
    particle_pos = liquid.get_particles_pos().cpu().numpy()
    # print("PARTICLE POS SHAPE:", particle_pos.shape)
    # print("PARTICLE POS:", particle_pos)
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

# ------------- move circularly to stir ----------------
def stir(scene, franka):
    # lower the rod into the liquid
    end_effector = franka.get_link('hand')
    n_move_steps = 50

    center = np.array([0.5, 0.0, 0.4])   # choose a center in world coordinates
    radius = 0.05                         # circle radius
    start_angle = 0.0
    end_angle   = 2 * np.pi               # full circle; change for arc


    for i in range(n_move_steps):
        alpha = i / n_move_steps
        theta = start_angle + alpha * (end_angle - start_angle)
    
        # Circle in XY plane around `center`, constant Z
        intermediate_pos = np.array([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            center[2],                     # keep height fixed
        ])

        q_move = franka.inverse_kinematics(
            link=end_effector,
            pos=intermediate_pos,
            quat=side_quat,
        )

        franka.control_dofs_position(q_move[:-2], motors_dof)
        scene.step()
    
    for _ in range(30):
        scene.step()

#-----------------------------------------------------------------------
#   COMPUTER VISION FUNCTIONS
#-----------------------------------------------------------------------
'''
Camera Calibrations:
INTRINSIC MATRIX 
[[277.12812921   0.         160.        ]
 [  0.         277.12812921 160.        ]
 [  0.           0.           1.        ]]
TRANSFORMATION MATRIX 
[[ 1.    0.    0.    0.65]
 [ 0.    0.   -1.   -0.5 ]
 [ 0.    1.    0.    0.  ]
 [ 0.    0.    0.    1.  ]]


Execution pipeline:
media_pipe creates a bouding box around the cup and gives the coordiates of the box
we find the center of the box or the bottom of the box (1x3 image coordinate)
we input the 1x3 image coordinates into this function, which returns 1x3 world coordinates
we input the world coordinates into the approach function
'''

'''
TODO: update this function to use mediapipe to get centers of left and right cup
'''
def get_cup_centers(cam):
    rgb_arr, depth_arr, seg_arr, normal_arr = cam.render(depth=True)

    # print("RGB", rgb_arr[0])
    # print("DEPTH", depth_arr[0])
    # print("DEPTH ARR SHAPE", depth_arr.shape

    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2GRAY)
    plt.plot(75, 130, 'ro') # replace this with predicted left center
    plt.plot(243, 130, 'ro') # replace this with predicted right center
    plt.imshow(gray)
    plt.show()

    return [(75, 130), (243, 130)]
    #-----------------------------------------------------------------------------
    # METHOD 1: using countours from CV2
    # (this is troublesome cause some extra shapes are countoured e.g the rod)
    # _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) #try wo inv.
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Make a copy of the original image to draw on
    # img_contours = rgb_arr.copy()  # or gray.copy() if you want grayscale background
    # # Draw all contours in green with thickness 2
    # cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    # # If using matplotlib, convert BGR to RGB
    # img_contours_rgb = cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)
    # # Show the image with contours
    # plt.figure(figsize=(8,6))
    # plt.imshow(img_contours_rgb)
    # plt.axis('off')
    # plt.show() 

    # cup_centers_px = []
    # for cnt in contours:
    #     M = cv2.moments(cnt)
    #     if M["m00"] > 0:
    #         cx = int(M["m10"] / M["m00"])
    #         cy = int(M["m01"] / M["m00"])
    #         cup_centers_px.append((cx, cy))
    # print("CUP CENTERS PX:", cup_centers_px)

    # plt.imshow(rgb_arr)
    # plt.axis('off')
    # plt.show()
    # pc, mask_arr = cam.render_pointcloud()
    #-----------------------------------------------------------------------------

'''
TODO: this is to find extrinsic matrix but it's currently not current
this funtion returns R: 
[[-1, 0, 0]
 [ 0, 1, 0]
 [ 0, 0, 1]]
should be:
[[ 1.,  0.,  0.], 
 [ 0.,  0., -1.],
 [ 0.,  1.,  0.]]
'''
def look_at_transform(pos, lookat, up):
    '''
    Generate the extrinsic matrix (world to camera)
    TODO: adjust so that lookat is at CUP3 (the one that's not moving)
    '''
    pos = np.array(pos)
    lookat = np.array(lookat)
    up = np.array(up)

    # forward vector (camera points toward +Y)
    f = lookat - pos # y axis
    f = f / np.linalg.norm(f)

    # left vector  (-X direction)
    l = np.cross(f, up) 
    l = l / np.linalg.norm(l)

    # recompute true up vector
    u = np.cross(l, f)

    # Genesis cameras look down -Z
    T = np.eye(4)
    T[:3, 0] = -l       # X axis
    T[:3, 1] = f       # Y axis
    T[:3, 2] = u       # Z axis (camera forward is -Z)
    T[:3, 3] = pos     # position
    return T

camera_position = np.array([settings.CAM_POS[0], settings.CAM_POS[1], settings.CAM_POS[2]])
# camera_position = np.array([0.55, 0.5, 0.]) #x->x, y->-z, z->y
R_cam_to_world = np.array([
    [1,  0,  0],  # camera X → world X
    [0,  0, 1],  # camera Y → world -Z
    [0,  1,  0]   # camera Z → world Y
])
def pixel_to_world(K, u, v, depth=1.0):
    """
    Transform pixel coordinates to world coordinates.
    
    Args:
        u, v: pixel coordinates (0 to 319)
        depth: depth value at that pixel (in meters)
    
    Returns:
        world_point: 3D point in world coordinates [x, y, z]
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    x_cam = (u - cx) * depth / fx
    y_cam = -(v - cy) * depth / fy
    z_cam = depth
    
    point_in_camera = np.array([x_cam, y_cam, z_cam])
    print("POINT IN CAMERA:", point_in_camera)
    
    point_in_world = R_cam_to_world @ point_in_camera + camera_position
    
    return point_in_world

def pour_drink(scene, franka, mixer_name, cup_entity, cam, pour_level, next_mixer_name=None):
    cups, depth = get_camera_render(cam)
    
    mixer_cup_x = cups[mixer_name][0][0]
    mixer_cup_y = cups[mixer_name][0][1]

    K = cam.intrinsics
    
    cup_pos = pixel_to_world(K, mixer_cup_x, mixer_cup_y)
    print(f"{mixer_name} CUP POS WORLD COORDINATES: {cup_pos}")
    print("TRUE LEFT CUP START POS", settings.LEFT_CUP_START_POS)
    print("TRUE RIGHT CUP START POS", settings.RIGHT_CUP_START_POS)
    approach(scene, franka, cup_pos)

    grasp(scene, franka)
    move_dist(scene, franka, 2, 0.02)
    move_dist(scene, franka, 1, 0.1)
    current_position = cup_entity.get_pos().cpu().numpy()
    lift(scene, franka, current_position, settings.LIFT_HEIGHT)
    dist_to_move = settings.POUR_LOCATION[0] - current_position[0]
    move_dist(scene, franka, 0, dist_to_move)

    rotate(scene, franka, pour_level, 1)
    move_dist(scene, franka, 0, -dist_to_move)
    lift(scene, franka, cup_entity.get_pos().cpu().numpy(), 0.08)
    ungrasp(scene, franka)
    move_dist(scene, franka, 1, 0.06)

    if next_mixer_name is not None:
        next_mixer_cup_x = cups[next_mixer_name][0][0]
        next_mixer_cup_y = cups[next_mixer_name][0][1]
        next_cup_pos = pixel_to_world(K, next_mixer_cup_x, next_mixer_cup_y)
        move_distance = next_cup_pos[0] - cup_entity.get_pos().cpu().numpy()[0]
        move_dist(scene, franka, 0, move_distance, 100)
