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
q0 = pin.neutral(model)
robot_viz.display(q0)

# 第四步，计算期望关节角度
loaded_qdes = np.load("qlog.npy") 
all_step = loaded_qdes.shape[0]
cur_step = 0
while True:
    if cur_step < all_step:
        ref_dof_pos = loaded_qdes[cur_step]
        print("id, dof_pos: ", cur_step, ", ", ref_dof_pos)
        q0[7:] = ref_dof_pos
        robot_viz.display(q0)
        cur_step += 1
    time.sleep(0.1)
