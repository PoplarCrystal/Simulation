import mujoco as mj
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import time
import os


class PendulumControl:
    def __init__(self, filename, is_show):
        # 1. model and data
        self.model = mj.MjModel.from_xml_path(filename)
        self.data = mj.MjData(self.model)
        self.is_show = is_show
        if self.is_show:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.keyboard_cb)
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.cam.lookat = [0.012768, -0.000000, 1.254336]
            self.viewer.cam.distance = 5.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -5
        # 2. init Controller
        self.ctrl_mode = "pos_mode"
        self.init_controller()

    def init_controller(self):
        # 1. set init pos
        self.data.qpos[0] = np.pi/2
        # 2. set the controller param
        # self.set_controller_param("pos_mode", 10.0, 1.0)
        # self.set_controller_param("vel_mode", 0, 1.0)
        self.set_controller_param("trq_mode", 0.0, 0.0)
        # 3. set the controller
        mj.set_mjcb_control(self.controller)

    def set_controller_param(self, ctrl_mode, kp, kv):
        if ctrl_mode == "trq_mode":
            self.ctrl_mode = ctrl_mode
            self.model.actuator_gainprm[0, 0] = 0
            self.model.actuator_biasprm[0, 1] = 0
            self.model.actuator_gainprm[1, 0] = 0
            self.model.actuator_biasprm[1, 2] = 0
        elif ctrl_mode == "pos_mode" or "vel_mode":
            self.ctrl_mode = ctrl_mode
            self.model.actuator_gainprm[0, 0] = kp
            self.model.actuator_biasprm[0, 1] = -kp
            self.model.actuator_gainprm[1, 0] = kv
            self.model.actuator_biasprm[1, 2] = -kv

    def controller(self, model, data):
        if self.ctrl_mode == "pos_mode":
            data.ctrl[0] = np.deg2rad(80.0)
        elif self.ctrl_mode == "vel_mode":
            data.ctrl[1] = 2.0
        elif self.ctrl_mode == "trq_mode":
            data.ctrl[2] = 10.0 * (np.deg2rad(110.0) - data.sensordata[0]) - 1.0 * data.sensordata[1]

    def main(self):
        sim_start, sim_end = time.time(), 100.0
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
    rel_path = "pendulum.xml"
    dir_name = os.path.dirname(__file__)
    xml_path = os.path.join(dir_name + "/" + rel_path)
    is_show = True

    pendulumControl = PendulumControl(xml_path, is_show)
    pendulumControl.main()
