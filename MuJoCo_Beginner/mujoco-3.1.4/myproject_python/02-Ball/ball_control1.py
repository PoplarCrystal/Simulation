import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os


class BallControl:
    def __init__(self, filename, is_show):
        # 1. model and data
        self.model = mj.MjModel.from_xml_path(filename)
        self.data = mj.MjData(self.model)
        self.is_show = is_show
        if self.is_show:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.keyboard_cb)
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.viewer.cam.lookat = [0.0, 0.0, 0.0]
            self.viewer.cam.distance = 8.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
        # 2. init Controller
        self.init_controller()

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = 0.1
        self.data.qvel[0] = 2.0
        self.data.qvel[2] = 5.0
        # 2. set the controller
        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
        """
        This controller adds drag force to the ball
        The drag force has the form of
        F = (cv^Tv)v / ||v||
        """
        vx, vy, vz = data.qvel[0], data.qvel[1], data.qvel[2]
        v = np.sqrt(vx * vx + vy * vy + vz * vz)
        c = 1.0
        data.qfrc_applied[0] = -c * v * vx
        data.qfrc_applied[1] = -c * v * vy
        data.qfrc_applied[2] = -c * v * vz

    def main(self):
        sim_start, sim_end = time.time(), 10.0
        while time.time() - sim_start < sim_end:
            step_start = time.time()
            loop_num, loop_count = 50, 0
            # 1. running for 0.002*50 = 0.1s
            while loop_count < loop_num:
                loop_count = loop_count + 1
                mj.mj_step(self.model, self.data)
            # 2. GUI show
            if self.is_show:
                if self.viewer.is_running():
                    self.viewer.cam.lookat[0] = self.data.qpos[0]
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
    rel_path = "ball.xml"
    dir_name = os.path.dirname(__file__)
    xml_path = os.path.join(dir_name + "/" + rel_path)
    is_show = False
    ballControl = BallControl(xml_path, is_show)

    ballControl.main()
