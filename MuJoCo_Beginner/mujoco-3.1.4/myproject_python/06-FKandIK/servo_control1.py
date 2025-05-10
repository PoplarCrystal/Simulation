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
            self.viewer.cam.lookat = [0.012768, -0.000000, 3.254336]
            self.viewer.cam.distance = 8.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -90
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
                writer.writerow(["time", "q1_ref", "q1", "q2_ref", "q2"])

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = -0.5
        self.data.qpos[1] = 1.0
        # 2. set the controller
        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        data.ctrl[0] = 0.7
        data.ctrl[2] = -0.3
        self.robot_fk()
        self.robot_ik()

    def robot_fk(self):
        """
            x = l1 * cos(q1) + l2 * cos(q1 + q2)
            y = l1 * sin(q1) + l2 * sin(q1 + q2)
        """
        q1 = self.data.qpos[0]
        q2 = self.data.qpos[1]
        l1, l2 = 1, 1
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        site_x = self.data.sensordata[0]
        site_y = self.data.sensordata[1]
        print("------------- fk ---------------")
        print("fk comp: ", x, y)
        print("sensor: ", site_x, site_y)

    def robot_ik(self):
        """
            r = sqrt(x^2 + y^2)
            cos(q2a) = (l1^2 + l2^2 - r^2) / (2 * l1 * l2)
            q2 = pi - q2a or -pi + q2a
            sin(q1a) = sin(q2a) / r * l2
            q1 = atan(y, x) - q1a
        """
        site_x = self.data.sensordata[0]
        site_y = self.data.sensordata[1]
        sensor_q1 = self.data.qpos[0]
        sensor_q2 = self.data.qpos[1]

        l1, l2 = 1, 1
        r = np.sqrt(site_x * site_x + site_y * site_y)
        cosq2a = min(1.0, max(-1.0, (l1 * l1 + l2 * l2 - r * r) / (2 * l1 * l2)))
        q2a1 = math.acos(cosq2a) - math.pi
        q2a2 = math.pi - math.acos(cosq2a)
        q2 = q2a1
        if math.fabs(q2a1 - sensor_q2) > math.fabs(q2a2 - sensor_q2):
            q2 = q2a2
        sinq1a = math.sin(math.pi - q2) / r * l2
        q1 = math.atan2(site_y, site_x) - math.asin(sinq1a)
        print("------------- ik ---------------")
        print("ik comp: ", q1, q2)
        print("sensor: ", sensor_q1, sensor_q2)

    # def generate_trajectory(self, t0, t1, x0, x1, t):
    #     """
    #         cubic trajectory: x(t) = a0 + a1*t + a2*t^2 + a3*t^3
    #         a0 = ( x0*t1^2*(t1-3*t0) + x1*t0^2*(3*t1-t0) )/ (t1-t0)^3
    #         a1 = 6*t0*t1*(x0-x1) / (t1-t0)^3
    #         a2 = 3*(t0+t1)*(x1-x0) / (t1-t0)^3
    #         a3 = 2*(x0-x1) / (t1-t0)^3
    #     """
    #     rate = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    #     a0 = x0
    #     a1 = 0
    #     a2 = 3 * (x1 - x0)
    #     a3 = 2 * (x0 - x1)
    #     x = a0 + a1 * rate + a2 * (rate ** 2) + a3 * (rate ** 3)
    #     dx = a1 + 2 * a2 * rate + 3 * a3 * (rate ** 2)
    #     return np.array([x, dx])

    def save_data(self, q1, q2):
        self.data_count = self.data_count + 1
        if self.data_count % self.data_freq != 0:
            return
        with open(self.csv_path, "a+", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.data.time, q1, self.data.qpos[0], q2, self.data.qpos[1]])
            # writer.writerow([self.data.time, q1, q2])

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
    is_save_data = False

    dbpendulumControl = DBPendulumControl(xml_path, is_show, is_save_data)
    dbpendulumControl.main()
