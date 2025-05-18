
import path
from base.base_mujoco import MuJoCoBase
from ur5_config import ConfigUR5
import numpy as np


class MuJoCoUR5(MuJoCoBase):
    def __init__(self, cfg: ConfigUR5):
        super().__init__(cfg)


    def pre_step(self):
        key_name = "home"
        pos_des = self.model.key("home").qpos
        vel_des = np.zeros_like(pos_des)
        tau_ff = np.zeros_like(pos_des)
        kp = np.diag([20, 200, 200, 200, 20, 20])
        kd = np.diag([2, 20, 20, 20, 2, 2])
        self.pvt_control(pos_des, vel_des, tau_ff, kp, kd)


    def pvt_control(self, pos_des, vel_des, trq_ff, kp, kd):
        self.data.ctrl = kp @ (pos_des - self.data.qpos) + kd @ (vel_des - self.data.qvel) + trq_ff



if __name__ == "__main__":
    ur5cfg = ConfigUR5()
    ur5 = MuJoCoUR5(ur5cfg)
    ur5.simulation()