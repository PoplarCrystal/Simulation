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
from scipy.spatial.transform import Rotation



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
        # self.mocap_id = self.model.body("target").id
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
        jac = np.zeros((6, self.model.nv))
        error = np.zeros(6)
        error_pos = error[:3]
        error_ori = error[3:]
        target_ori = np.zeros(4)
        site_quat = np.zeros(4)
        target_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        radius, cx, cy = 0.15, 0.5, 0.0
        frequency = 0.5
        self.data.mocap_pos[self.mocap_id][0:2] = mujoco_utils.curve(self.data.time, radius, cx, cy, frequency)

        error_pos[:] = self.data.site(self.site_id).xpos - self.data.mocap_pos[self.mocap_id]
    
        target_ori = self.data.mocap_quat[self.mocap_id]
        mujoco.mju_negQuat(target_quat_conj, target_ori)
        mujoco.mju_mat2Quat(site_quat, self.data.site(self.site_id).xmat)
        mujoco.mju_mulQuat(error_quat, site_quat, target_quat_conj)
        # Convert error quaternion to axis-angle representation
        # We do so, as the Jacobian function we will use represent orientation error in axis-angle form
        mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
        
        # Get the Jacobian with respect to the end-effector site.
        # This function calculate the Jacobian of the world coordinates of a body frame
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], self.site_id)

        # Solve the differential IK
        # We want to have the error equal to zero
        # We take a step dq such J dq = -error 
        # Note, the origin differential IK works on J v = -speed * error / dt
        # Here we implement a simple version by using dq and making Jdq = -error.
        dq = np.linalg.pinv(jac) @ -error

        # Our robot arm is position controlled, so we simple give it the target joint configure
        q = self.data.qpos.copy()
        # Add dq to q, here results should be the same as q = q + dq. It is different when q includes quaternian
        mujoco.mj_integratePos(self.model, q, dq, 1)
    
        # # Our robot is configured to be position control
        # # Here we direct set the control signal to the desired position
        np.clip(q, *self.model.jnt_range.T, out=q)
        self.data.ctrl = q

    def post_step(self):
        """ 增加发布信息 """
        msg = franka_states_t()
        msg.timestamp = int(time.time() * 1e9)
        msg.joint_cmd = self.joint_cmd
        msg.joint_state = self.joint_state
        msg.ee_pos_cmd = self.ee_pos_cmd
        msg.ee_pos_state = self.ee_pos_state
        msg.ee_quat_cmd = self.ee_quat_cmd
        msg.ee_quat_state = self.ee_quat_state
        self.lcm.pub_msg(msg)

    # def viewer_step(self):
        """ 画图 """
        self.target_traj.append(self.data.mocap_pos[self.mocap_id].copy())
        self.end_effector_traj.append(self.data.site(self.site_id).xpos.copy())
        mujoco_utils.modify_scene(self.viewer, self.target_traj[::10], self.end_effector_traj[::10])

if __name__ == "__main__":
    frankaConfig = ConfigFranka()
    frankaRobot = MuJoCoFranka(frankaConfig)
    frankaRobot.simulation()
