import mujoco as mj
import mujoco.viewer
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are
import control as ct
import time
import os
import csv


class LQRControl:
    def __init__(self, filename, is_show, is_save_data):
        # 1. model and data
        self.model = mj.MjModel.from_xml_path(filename)
        self.data = mj.MjData(self.model)
        self.is_show = is_show
        self.is_save_data = is_save_data
        if self.is_show:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.keyboard_cb)
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.cam.lookat = [0.012768, -0.000000, 1.254336]
            self.viewer.cam.distance = 10.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -5
        # 2. init Controller
        self.init_controller()
        # 3. save data
        if self.is_save_data:
            self.data_count = 0
            self.data_freq = 50   # timestep(2ms) * 50 = 100ms
            self.csv_path = "demo.csv"
            with open(self.csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(["time", "q1", "q2"])

    def init_controller(self):
        # 1. 求解控制器参数
        self.get_control_param()
        # 2. 设置初始条件
        self.data.qpos[0] = np.deg2rad(1)
        mj.mj_forward(self.model, self.data)
        # 3. 设置控制器
        mj.set_mjcb_control(self.controller)

    def get_dynamics(self, inputs):
        """
            Dynamics Equation: M*ddq + fbias = tau
                ddq = M^(-1) * (tau - fbias)

            state: x = [q1, dq1, q2, dq2]
            input: u
            output: y = [dq1, ddq1, dq2, ddq2]
        """
        # 1. mj_forward
        self.data.qpos[0] = inputs[0]
        self.data.qvel[0] = inputs[1]
        self.data.qpos[1] = inputs[2]
        self.data.qvel[1] = inputs[3]
        self.data.ctrl[0] = inputs[4]
        mj.mj_forward(self.model, self.data)
        # 2. data process
        dq1 = self.data.qvel[0]
        dq2 = self.data.qvel[1]
        M = np.zeros((2, 2))
        mj.mj_fullM(self.model, M, self.data.qM)
        f = np.array([0 - self.data.qfrc_bias[0], self.data.ctrl[0] - self.data.qfrc_bias[1]])
        ddq = inv(M) @ f
        return np.array([dq1, ddq[0], dq2, ddq[1]])

    def get_control_param(self, pert=0.001):
        """
            Linearization: dx = Ax + Bu
            dx = f(q1, dq1, q2, dq2, u)
            A: 4x4
            B: 4x1
            x: 4
        """
        # 1. Linearization
        f0 = self.get_dynamics(np.zeros(5))
        Jacobians = []
        for i in range(5):
            inputs_i = np.zeros(5)
            inputs_i[i] = pert
            jac = (self.get_dynamics(inputs_i) - f0) / pert
            Jacobians.append(jac[:, np.newaxis])
        A = np.concatenate(Jacobians[:4], axis=1)
        B = Jacobians[-1]
        # 2. Param
        Q = np.diag([1, 1, 1, 1])
        R = np.diag([1])
        self.K, _, _ = ct.lqr(A, B, Q, R)

    def controller(self, model, data):
        """
            This function implements a LQR controller for balancing.
            u = -K(x-xd)
        """
        state = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
        data.ctrl[0] = (-self.K @ state)[0]

        # Apply noise to shoulder
        # noise = mj.mju_standardNormal(0.0)
        # data.qfrc_applied[0] = noise

        if self.is_save_data:
            self.save_data()

    def save_data(self):
        self.data_count = self.data_count + 1
        if self.data_count % self.data_freq != 0:
            return
        with open(self.csv_path, "a+", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.data.time, self.data.qpos[0], self.data.qpos[1]])

    def main(self):
        sim_start, sim_end = time.time(), 100.0
        while time.time() - sim_start < sim_end:
            step_start = time.time()
            loop_num, loop_count = 20, 0
            # 1. running for 0.002*50 = 0.1s
            while loop_count < loop_num:
                loop_count = loop_count + 1
                mj.mj_step(self.model, self.data)
            # 2. GUI show
            if self.is_show:
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    break
            # 3. sleep for next period
            step_next_delta = self.model.opt.timestep * loop_count - (time.time() - step_start)
            if step_next_delta > 0:
                time.sleep(step_next_delta)
        if self.is_show:
            self.viewer.close()

    def keyboard_cb(self, keycode):
        if chr(keycode) == ' ':
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.init_controller()


if __name__ == "__main__":
    rel_path = "db_pendulum.xml"
    dir_name = os.path.dirname(__file__)
    xml_path = os.path.join(dir_name + "/" + rel_path)
    is_show = True
    is_save_data = True

    lqrControl = LQRControl(xml_path, is_show, is_save_data)
    lqrControl.main()
