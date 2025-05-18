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
pin.updateFramePlacements(fmodel, fdata)

# 第三步，得到坐标变换关系
frame_id = fmodel.getFrameId("right_ankle_roll_joint")
oMf = fdata.oMf[frame_id].copy()
R = oMf.rotation
p = oMf.translation
homo = oMf.homogeneous
homo1 = np.identity(4)
homo1[:3, :3] = R
homo1[:3, 3] = p
print("------ homogeneous matrix ------")
print(homo)
print(homo1)
action = oMf.action
action1 = np.identity(6)
action1[:3, :3] = R
action1[:3, 3:] = pin.skew(p) @ R
action1[3:, 3:] = R
print("------ action matrix ------")
print(action)
print(action1)

# 第四步，得到三个坐标系之间的关系
Vb = pin.getFrameVelocity(fmodel, fdata, frame_id, pin.ReferenceFrame.LOCAL)
Vs = pin.getFrameVelocity(fmodel, fdata, frame_id, pin.ReferenceFrame.WORLD)
Va = pin.getFrameVelocity(fmodel, fdata, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

Vs1 = oMf.act(Vb)
Vs2 = oMf.action @ Vb
T = np.identity(6)
T[:3, :3] = R
T[3:, 3:] = R
Va1 = T @ Vb
print("------ Vs ------")
print(Vs.vector)
print(Vs1.vector)
print(Vs2)
print("------ Va ------")
print(Va.vector)
print(Va1)
