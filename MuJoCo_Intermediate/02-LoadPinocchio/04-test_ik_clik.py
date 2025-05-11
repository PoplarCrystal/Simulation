import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
from conf import Conf

# Reference: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b_examples_d_inverse_kinematics.html

# 第一步，导入模型
fmodel = pin.buildModelFromUrdf(Conf.urdf_robot_filename)
fdata = fmodel.createData()

# 第二步，关节赋值，计算正运动学
fq_zero = pin.neutral(fmodel)
# 将左腿的三个Hip关节进行赋值，得到一个屈膝的位姿，将其作为期望位姿
q_des = fq_zero.copy()
q_des[0] = -0.2
q_des[1] = 0.1
q_des[2] = -0.3
q_des[3] = 0.5
q_des[4] = -0.3
q_des[5] = 0.1
joint_id = fmodel.getJointId("left_ankle_roll_joint")
pin.forwardKinematics(fmodel, fdata, q_des)
oMi_des = fdata.oMi[joint_id].copy()

# 第三步，求解逆运动学
# 逆运动学求解的参数
eps = 1e-4
ITER_MAX = 1000
DT = 1e-1
damp = 1e-3
success = False
q_guess = fq_zero
q_guess[3] = 0.1  # 正向屈膝，作为先验条件

# 开始求解逆运动学
q_log = []
q = q_guess
for i in range(ITER_MAX):
    pin.forwardKinematics(fmodel, fdata, q)
    iMd = fdata.oMi[joint_id].actInv(oMi_des)
    err = pin.log(iMd).vector  # in joint frame
    if np.linalg.norm(err) < eps:
        success = True
        print('Convergence achieved times: ', i)
        break
    J = pin.computeJointJacobian(fmodel, fdata, q, joint_id)  # in joint frame
    J = -np.dot(pin.Jlog6(iMd.inverse()), J)  # 修正为err的梯度(雅可比矩阵)

    v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pin.integrate(fmodel, q, v * DT)
    q_log.append(q)
    if not i % 10:
        print(f"{i}: error = {err.T}")

if success:
    print("Convergence achieved!")
    np.save("qlog.npy", np.vstack(q_log))
else:
    print(
        "\n"
        "Warning: the iterative algorithm has not reached convergence "
        "to the desired precision"
    )
print(f"\ndesired: {q_des.flatten().tolist()}")
print(f"\nresult: {q.flatten().tolist()}")
print(f"\nfinal error: {err.T}")