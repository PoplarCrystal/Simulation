import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import math
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
            self.data_freq = 10  # timestep(2ms) * 50 = 100ms
            self.csv_path = "demo.csv"
            with open(self.csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(["time", "des_x", "x", "des_z", "z"])

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = -0.5
        self.data.qpos[1] = 1.0
        mj.mj_forward(self.model, self.data)
        radius = 0.5
        self.circle_param = [self.data.sensordata[0] - radius, self.data.sensordata[2], radius]
        # 2. set the controller
        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        x0, z0, r = self.circle_param[0], self.circle_param[1], self.circle_param[2]
        cart_des = [x0 + r * math.cos(data.time), z0 + r * math.sin(data.time)]
        joint_des = self.robot_ik(cart_des)

        data.ctrl[0] = joint_des[0, 0]
        data.ctrl[2] = joint_des[1, 0]

        if self.is_save_data:
            self.save_data(cart_des[0], cart_des[1])

    def robot_ik(self, cart_pos):
        """
            This function implements a P controller for tracking
            the reference motion.
            Δq = Jinv * Δx
            calculate Jacobian Matrix
            x = l1 * cos(q1) + l2 * cos(q1 + q2)
            z = l1 * sin(q1) + l2 * sin(q1 + q2)

            dx = (-l1 * sin(q1) - l2 * sin(q1 + q2)) * dq1 + (-l2 * sin(q1 + q2)) * dq2
            dz = (l1 * cos(q1) + l2 * cos(q1 + q2)) * dq1 + (l2 * cos(q1 + q2)) * dq2

            J = [-l1 * sin(q1) - l2 * sin(q1 + q2), -l2 * sin(q1 + q2)]
                [l1 * cos(q1) + l2 * cos(q1 + q2),  l2 * cos(q1 + q2)]

        """
        # End-effector position
        # ee_pos = self.data.sensordata[:3]
        # jacp = np.zeros((3, 2))
        # mj.mj_jac(self.model, self.data, jacp, None, ee_pos, 2)
        # J = jacp[[0, 2], :]
        # delta_pos = np.array([[cart_pos[0] - ee_pos[0]],
        #                       [cart_pos[1] - ee_pos[2]]])

        ee_pos = [self.data.sensordata[0], self.data.sensordata[2]]
        sinq1, cosq1 = math.sin(self.data.qpos[0]), math.cos(self.data.qpos[0])
        sinq12, cosq12 = math.sin(self.data.qpos[0] + self.data.qpos[1]), math.cos(self.data.qpos[0] + self.data.qpos[1])
        l1, l2 = 1, 1
        J = np.array([[-l1 * sinq1 - l2 * sinq12, -l2 * sinq12],
                      [l1 * cosq1 + l2 * cosq12,  l2 * cosq12]])
        delta_pos = np.array([[cart_pos[0] - ee_pos[0]],
                              [cart_pos[1] - ee_pos[1]]])
        dq = np.linalg.pinv(J) @ delta_pos
        return np.array([[self.data.qpos[0] + dq[0, 0]],
                         [self.data.qpos[1] + dq[1, 0]]])

    def generate_trajectory(self, t0, t1, x0, x1, t):
        """
            cubic trajectory: x(t) = a0 + a1*t + a2*t^2 + a3*t^3
            a0 = ( x0*t1^2*(t1-3*t0) + x1*t0^2*(3*t1-t0) )/ (t1-t0)^3
            a1 = 6*t0*t1*(x0-x1) / (t1-t0)^3
            a2 = 3*(t0+t1)*(x1-x0) / (t1-t0)^3
            a3 = 2*(x0-x1) / (t1-t0)^3
        """
        rate = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        a0 = x0
        a1 = 0
        a2 = 3 * (x1 - x0)
        a3 = 2 * (x0 - x1)
        x = a0 + a1 * rate + a2 * (rate ** 2) + a3 * (rate ** 3)
        dx = a1 + 2 * a2 * rate + 3 * a3 * (rate ** 2)
        return np.array([x, dx])

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
