import path
MUJOCO_DIR = path.project_root
# MuJoCo相关的Module
import mujoco
import mujoco as mj
from base.base_mujoco import MuJoCoBase
from franka_config import ConfigFranka
import utils.mujoco_utils as mujoco_utils
# Pinocchio相关的Module
from utils.pino_solver import PinoSolver
import pinocchio as pin
# LCM通信相关的Module
from utils.lcm_pub import LCM
from lcm_types.franka.franka_states_t import franka_states_t
# 路径相关
import os
import time
import numpy as np



class MuJoCoFranka(MuJoCoBase):
    def __init__(self, cfg: ConfigFranka):
        super().__init__(cfg)
        """ 重置机器人的位姿 """
        self.viewer.opt.sitegroup[4] = 1
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
        mj.mj_forward(self.model, self.data)
        mjcf_path = MUJOCO_DIR + "/models/franka_emika_panda/panda_nohand.xml"
        self.pino_solver = PinoSolver(mjcf_path, xml_type="MJCF")
        # 这里两者都可以从MuJoCo中获取，但是实际上，真机不可能获取末端位姿
        self.ee_frame_id = self.pino_solver.model.getFrameId("attachment")
        self.mocap_id = self.model.body("target").mocapid[0]
        self.site_id = self.model.site("attachment_site").id
        self.lcm = LCM("franka_demo")
        # 控制相关的参数
        self.joint_cmd = np.zeros(7)
        self.joint_state = np.zeros(7)
        self.ee_pos_cmd = np.zeros(3)
        self.ee_quat_cmd = np.zeros(4)
        self.ee_pos_state = np.zeros(3)
        self.ee_quat_state = np.zeros(4)
        # 画图相关
        self.target_traj = []
        self.end_effector_traj = []
        self.frames = 0
        self.framerate = 30
        

    def pre_step(self):
        """ Step1: 计算期望 """
        # 设置mocap的轨迹，可以程序设置，也可以自己拖动设置, 作为期望的末端位置
        drag_flag = 0
        if not drag_flag:
            radius, cx, cy = 0.15, 0.5, 0.0
            frequency = 0.5
            # self.data.body(self.mocap_id).xpos[0:2] = mujoco_utils.curve(self.data.time, radius, cx, cy, frequency)
            self.data.mocap_pos[self.mocap_id][0:2] = mujoco_utils.curve(self.data.time, radius, cx, cy, frequency)
        self.ee_pos_cmd[:] = self.data.mocap_pos[self.mocap_id][:]
        self.ee_quat_cmd = pin.Quaternion(self.data.mocap_quat[self.mocap_id][[1, 2, 3, 0]])
        ee_mat_cmd = self.ee_quat_cmd.toRotationMatrix()

        # """ Step2: 计算当前状态 """
        self.joint_state = self.data.qpos.copy()
        self.pino_solver.update_kin_dyn(self.joint_state)
        ee_pose = self.pino_solver.get_frame_pose(self.ee_frame_id)
        self.ee_pos_state = ee_pose.translation
        self.ee_quat_state = pin.Quaternion(ee_pose.rotation)
        ee_mat_state = ee_pose.rotation
        ee_jac = self.pino_solver.get_frame_jac(self.ee_frame_id)

        # # """ Step3: 计算误差"""
        # 两种方式: des - cur 或者是cur - des
        # 在这里，我们采用des - cur
        error_pos = self.ee_pos_cmd - self.ee_pos_state
        error_ori = ee_mat_state @ pin.log3(ee_mat_state.transpose() @ ee_mat_cmd)
        error = np.hstack((error_pos, error_ori))

        # """ Step4: 做控制 """
        dq = np.linalg.pinv(ee_jac) @ error
        q = self.joint_state.copy()
        # # 不涉及浮动基座，可以直接求和, 如果涉及浮动基座，需要单独求
        q = pin.integrate(self.pino_solver.model, q, dq)
        self.data.ctrl[:] = q

    def post_step(self):
        """ 增加发布信息 """
        msg = franka_states_t()
        msg.timestamp = int(time.time() * 1e9)
        msg.joint_cmd = self.joint_cmd
        msg.joint_state = self.joint_state
        msg.ee_pos_cmd = self.ee_pos_cmd
        msg.ee_pos_state = self.ee_pos_state
        msg.ee_quat_cmd = self.ee_quat_cmd.coeffs()
        msg.ee_quat_state = self.ee_quat_state.coeffs()
        self.lcm.pub_msg(msg)

        """ 画图 """
        self.target_traj.append(self.data.mocap_pos[self.mocap_id].copy())
        self.end_effector_traj.append(self.data.site(self.site_id).xpos.copy())
        mujoco_utils.modify_scene(self.viewer, self.target_traj[::10], self.end_effector_traj[::10])

if __name__ == "__main__":
    frankaConfig = ConfigFranka()
    frankaRobot = MuJoCoFranka(frankaConfig)
    frankaRobot.simulation()
