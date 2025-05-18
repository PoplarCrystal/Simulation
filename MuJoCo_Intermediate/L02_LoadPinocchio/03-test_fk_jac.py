import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
from conf import Conf

# 第一步，导入模型
fmodel = pin.buildModelFromUrdf(Conf.urdf_robot_filename, pin.JointModelFreeFlyer())
fdata = fmodel.createData()

# 第二步，关节赋值，计算正运动学
# fq = pin.neutral(fmodel)
fq = pin.randomConfiguration(fmodel)
# fq的前7个元素是浮动基座的元素，不能使用random，需要单独赋值
qf_pose = pin.SE3.Random()
fq[:3] = qf_pose.translation
fq[3:7] = pin.Quaternion(qf_pose.rotation).coeffs()
fdq = np.random.randn(fmodel.nv)

pin.forwardKinematics(fmodel, fdata, fq, fdq)
pin.computeJointJacobians(fmodel, fdata, fq)
pin.updateFramePlacements(fmodel, fdata)

# 第三步，得到坐标变换关系
frame_id = fmodel.getFrameId("right_ankle_roll_joint")
"""
理解雅可比矩阵的三个坐标系
pin.ReferenceFrame.LOCAL(物体坐标系)
pin.ReferenceFrame.WORLD(空间坐标系)
pin.ReferenceFrame.LOCAL_WORLD_ALIGNED(物体系对齐空间系)
"""
################
## 1. 物体坐标系的对应关系
LOCAL_PARAMETER = (fmodel, fdata, frame_id, pin.ReferenceFrame.LOCAL)
Vb = pin.getFrameVelocity(*LOCAL_PARAMETER).vector
Jb = pin.getFrameJacobian(*LOCAL_PARAMETER)
Vb1 = Jb @ fdq
print('Vb: -----------------------')
print(Vb)
print(Vb1)

## 2. 空间坐标系的对应关系
WORLD_PARAMETER = (fmodel, fdata, frame_id, pin.ReferenceFrame.WORLD)
Vs = pin.getFrameVelocity(*WORLD_PARAMETER).vector
Js = pin.getFrameJacobian(*WORLD_PARAMETER)
Vs1 = Js @ fdq
print('Vs: -----------------------')
print(Vs)
print(Vs1)

## 3. 物体系对齐空间系
LOWOA_PARAMETER = (fmodel, fdata, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
Va = pin.getFrameVelocity(*LOWOA_PARAMETER).vector
Ja = pin.getFrameJacobian(*LOWOA_PARAMETER)
Va1 = Ja @ fdq
print('Va: -----------------------')
print(Va)
print(Va1)