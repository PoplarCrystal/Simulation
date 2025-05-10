import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os
import csv


class DBPendulumControl:
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
        # 1. set init pos
        self.data.qpos[0] = np.deg2rad(10)
        # 2. set the controller
        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        """
            1. Dynamics Equation
                (1) Newton-Euler Equation: M*dqq + C + G = t_in + t_out
                (2) Mujoco Equation: M*qacc + qfrc_bias = qfrc_applied + ctrl

            2. Model-Based Controller
                (1) PD Controller:
                    tau = Kp * (q_ref - q) - Kd * dq
                (2) PD Controller + Feedforward comp(gravity + coriolis forces):
                    tau = Kp * (q_ref - q) - Kd * dq + C + G
                (3) Feedback linearization
                    tau = M * (Kp * (q_ref - q) - Kd * dq) + C + G

            3. Data Source
                (1) Mujoco
                    data.q, data.qvel, data.qM, data.qfrc_bias ...
                (2) Other Dynamics Library(KDL, RBDL, Drake, Pinocchio ...)

        """
        M = np.zeros((2, 2))
        mj.mj_fullM(model, M, data.qM)
        f = data.qfrc_bias

        Kp = 100 * np.eye(2)
        Kd = 10 * np.eye(2)
        qref = np.array([-0.5, -1.6])

        ctrl1 = Kp @ (qref - data.qpos) - Kd @ data.qvel
        ctrl2 = f
        ctrl3 = M @ ctrl1
        # 1. PD Controller
        # data.qfrc_applied = ctrl1
        # 2. PD + Feedforward
        data.qfrc_applied = ctrl1 + ctrl2
        # 3. Feedback + Linearization
        # data.qfrc_applied = ctrl3 + ctrl2

        if self.is_save_data:
            self.save_data()

    def save_data(self):
        self.data_count = self.data_count + 1
        if self.data_count % self.data_freq != 0:
            return
        with open(self.csv_path, "a+", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.data.time, self.data.qpos[0], self.data.qpos[1]])
        print(self.data.time)

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
    is_save_data = False

    dbpendulumControl = DBPendulumControl(xml_path, is_show, is_save_data)
    dbpendulumControl.main()
