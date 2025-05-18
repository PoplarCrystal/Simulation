import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
from conf import Conf

np.set_printoptions(precision=4, suppress=True)

# 第一步，导入模型
fmodel = pin.buildModelFromUrdf(Conf.urdf_robot_filename, pin.JointModelFreeFlyer())
fdata = fmodel.createData()

# 第二步，关节赋值
# fq = pin.neutral(fmodel)
fq = pin.randomConfiguration(fmodel)
# fq的前7个元素是浮动基座的元素，不能使用random，需要单独赋值
qf_pose = pin.SE3.Random()
fq[:3] = qf_pose.translation
fq[3:7] = pin.Quaternion(qf_pose.rotation).coeffs()
fdq = np.random.randn(fmodel.nv)
fddq = np.random.randn(fmodel.nv)

print("--------------------------------")
tau = pin.rnea(fmodel, fdata, fq, fdq, fddq)
print("tau: ", tau)
G = pin.computeGeneralizedGravity(fmodel, fdata, fq)
print("G: ", G)
C = pin.computeCoriolisMatrix(fmodel, fdata, fq, fdq)
print("C: ", C) 
M = pin.crba(fmodel, fdata, fq)
print("M: ", M)

tau1 = M @ fddq + C @ fdq + G
print("tau_diff: ", tau - tau1)

