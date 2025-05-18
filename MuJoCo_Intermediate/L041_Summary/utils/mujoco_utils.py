import mujoco
import numpy as np


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.user_scn.ngeom >= scene.user_scn.maxgeom:
        scene.user_scn.ngeom = 0  # user_scn不会自动清除geom
        # return
    scene.user_scn.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.user_scn.geoms[scene.user_scn.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.user_scn.geoms[scene.user_scn.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1, point2)

"""
    用于向场景里面增加轨迹
"""  
def modify_scene(scn, target_traj, end_effector_traj):
    """Draw position trace"""
    if len(target_traj) > 1:
        for i in range(len(target_traj)-1):
            add_visual_capsule(scn, target_traj[i], target_traj[i+1], 0.005, np.array([0, 0, 1.0, 1.0]))
            add_visual_capsule(scn, end_effector_traj[i], end_effector_traj[i+1], 0.005, np.array([1.0, 0, 0, 0.8]))

"""
    画心形线或者圆
"""
def curve(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    theta = 2 * np.pi * f * t
    ## 画圆
    # x = r * np.cos(theta) + h
    # y = r * np.sin(theta) + k
    
    ## 画心形线
    r = r / 16
    x = r * (16 * np.sin(theta)**3) + h
    y = r * (13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)) + k
    return np.array([x, y])