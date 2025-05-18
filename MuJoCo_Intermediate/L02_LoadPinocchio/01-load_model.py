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

pin.forwardKinematics(fmodel, fdata, fq)
pin.updateFramePlacements(fmodel, fdata)

# 第三步，打印joint运动树与frame运动树
print("joint: {:d} -------------------------".format(fmodel.njoints))
for idx, name, oMi in zip(range(fmodel.njoints), fmodel.names, fdata.oMi):
    print("id {:<3} {:<24} : {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f}".format(
        idx, name, *oMi.translation.T.flat, *pin.rpy.matrixToRpy(oMi.rotation)))
print("frame: {:d} -------------------------".format(fmodel.nframes))
for idx, frame, oMf in zip(range(fmodel.nframes), fmodel.frames, fdata.oMf):
    print("id {:<3} {:<24} : {: .4f} {: .4f} {: .4f} {: .4f} {: .4f} {: .4f}".format(
        idx, frame.name, *oMf.translation.T.flat, *pin.rpy.matrixToRpy(oMf.rotation)))

