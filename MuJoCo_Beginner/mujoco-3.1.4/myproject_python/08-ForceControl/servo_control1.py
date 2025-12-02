import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import math
import time
import os
import csv
from realtime_plot import Plot

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
        self.plot = Plot()
        if self.is_save_data:
            self.data_count = 0
            self.data_freq = 10  # timestep(2ms) * 50 = 100ms
            self.csv_path = "demo.csv"
            with open(self.csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(["time", "des_x", "x", "des_z", "z"])

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = 0.8
        self.data.qpos[1] = -1.6
        mj.mj_forward(self.model, self.data)
        self.control_param = [self.data.sensordata[0], self.data.sensordata[2]]
        # 2. set the controller
        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        """
        1. Joint Impedance Control
            Ref to 04

        2. Cartesian Impedance Control
            tau = J^T*( Kd*(Xd-X)-Bd*X - Md*dJ*q) + C + G
        """
        cart_des = self.control_param
        if 5 < self.data.time < 10:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
            data.xfrc_applied[body_id][0] = -60
        tau = self.cart_imp_control(cart_des)
        data.ctrl[0] = tau[0]
        data.ctrl[1] = tau[1]

        self.plot.update_data(data.sensordata[0], data.sensordata[2])
        if self.is_save_data:
            self.save_data(cart_des[0], cart_des[1])

    def cart_imp_control(self, cart_pos):
        """
            Md = J^(-T) * M * J^(-1)
            tau = J^T*( Kd*(Xd-X)-Bd*dX - Md*dJ*dq) + C + G
        """
        # 1. Jacobian and dot Jacobian
        q1, q2 = self.data.qpos[0], self.data.qpos[1]
        dq1, dq2 = self.data.qvel[0], self.data.qvel[1]
        sinq1, cosq1 = math.sin(q1), math.cos(q1)
        sinq12, cosq12 = math.sin(q1 + q2), math.cos(q1 + q2)
        l1, l2 = 1, 1
        J = np.array([[-l1 * sinq1 - l2 * sinq12, -l2 * sinq12],
                      [l1 * cosq1 + l2 * cosq12, l2 * cosq12]])
        dJ = np.array([[-l1 * cosq1 * dq1 - l2 * cosq12 * (dq1 + dq2), -l2 * cosq12 * (dq1 + dq2)],
                      [-l1 * sinq1 * dq1 - l2 * sinq12 * (dq1 + dq2), -l2 * sinq12 * (dq1 + dq2)]])
        # 2. J^(-1)， J^(T), J^(-T)
        Jinv = np.linalg.pinv(J)
        JT = J.transpose()
        JinvT = Jinv.transpose()
        # 3. M, C + G
        M = np.zeros((2, 2))
        mj.mj_fullM(self.model, M, self.data.qM)
        Fbias = self.data.qfrc_bias
        # 4. Md, Bd, Kd
        Md = JinvT @ M @ Jinv
        Bd = 300 * np.eye(2)
        Kd = 120 * np.eye(2)
        # print("--------------------")
        # print(J)
        # print(dJ)
        # print(Jinv)
        # print(JinvT)
        # print(JT)
        # 5. controller
        Xd = np.array([cart_pos[0], cart_pos[1]])
        X = np.array([self.data.sensordata[0], self.data.sensordata[2]])
        dX = np.array([self.data.sensordata[3], self.data.sensordata[5]])
        dq = np.array([dq1, dq2])
        tau = JT @ (Kd @ (Xd - X) - Bd @ dX - Md @ dJ @ dq) + Fbias
        # tau = Fbias
        return tau

    def save_data(self, data1, data2):
        self.data_count = self.data_count + 1
        if self.data_count % self.data_freq != 0:
            return
        with open(self.csv_path, "a+", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.data.time, data1, self.data.sensordata[0], data2, self.data.sensordata[2]])

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
                    # print(self.viewer.cam.lookat[0], self.viewer.cam.lookat[1], self.viewer.cam.lookat[2],
                    #       self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.azimuth)
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

    dbpendulumControl = DBPendulumControl(xml_path, is_show, is_save_data)
    dbpendulumControl.main()
