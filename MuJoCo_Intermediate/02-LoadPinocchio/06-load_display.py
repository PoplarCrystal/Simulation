import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
from conf import Conf
from common import deadzone


# 第一步，导入全模型(计算模型，可视化模型和碰撞模型)
robot_model = RobotWrapper.BuildFromURDF(Conf.urdf_robot_filename, Conf.urdf_robot_directory, pin.JointModelFreeFlyer())
model = robot_model.model
visual_model = robot_model.visual_model
collision_model = robot_model.collision_model 

# 第二步，导入Meshcat模型
rightFoot = 'right_ankle_roll_joint'  # foot link name 
leftFoot =  'left_ankle_roll_joint'
robot_viz = MeshcatVisualizer(model, collision_model, visual_model)
robot_viz.initViewer(open=True)
robot_viz.loadViewerModel(rootNodeName="g1")
frame_ids=[model.getFrameId(rightFoot), model.getFrameId(leftFoot)]
robot_viz.displayFrames(visibility=True, frame_ids=frame_ids)

# 第三步，展出初始状态
default_joint_angles =  np.array([-0.20, 0.0, 0.0, 0.58, -0.38, 0.0, 
                                  -0.20, 0.0, 0.0, 0.58, -0.38, 0.0])
q0 = pin.neutral(model)
q0[6] = 1  # q.w
q0[2] = 1 # z
q0[7:19] = default_joint_angles
robot_viz.display(q0)

# 第四步，计算期望关节角度
dt, cycle_time = 0.1, 0.8
phase = 0
def compute_q_des():
    ref_dof_pos = np.zeros(12)
    global dt, cycle_time, phase
    # Step1: 计算相位
    phase += dt / cycle_time 
    sin_pos = np.sin(2 * np.pi * phase)
    sin_pos_deadzone = deadzone(sin_pos, 0.1)
    sin_pos_l = -sin_pos_deadzone.copy()
    sin_pos_r = sin_pos_deadzone.copy() 

    # Step2: 计算幅度
    scale =  0.3 #  控制幅度大小
    scale_hip = 1.0 * scale
    scale_knee = 1.5 * scale # 模拟某种步态
    scale_ankle = 0.5* scale
    # left swing
    sin_pos_l = max(0.0, sin_pos_l)
    ref_dof_pos[0] = -sin_pos_l * scale_hip
    ref_dof_pos[3] = sin_pos_l * scale_knee
    ref_dof_pos[4] = -sin_pos_l * scale_ankle 
    # right swing
    sin_pos_r = max(0.0, sin_pos_r)
    ref_dof_pos[6] = -sin_pos_r * scale_hip
    ref_dof_pos[9] = sin_pos_r * scale_knee
    ref_dof_pos[10] = -sin_pos_r * scale_ankle
    # double support
    ref_dof_pos = 0 if  np.abs(sin_pos) < 0.05 else ref_dof_pos
    ref_dof_pos += default_joint_angles
    return ref_dof_pos

# 第五步，进行迈步测试
while True:
    ref_dof_pos = compute_q_des()
    q0[7:] = ref_dof_pos
    robot_viz.display(q0)
    time.sleep(dt)
