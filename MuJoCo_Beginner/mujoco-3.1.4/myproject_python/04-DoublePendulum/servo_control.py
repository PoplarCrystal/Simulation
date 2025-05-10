import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import csv

class DBPendulumControl:
    def __init__(self, xml_path, is_show, is_savedata):
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.is_show = is_show
        self.is_savedata = is_savedata

        # add opengl api
        if self.is_show:
            self.cam = mj.MjvCamera()
            self.opt = mj.MjvOption()
            # 1. Init GLFW
            glfw.init()

            self.window = glfw.create_window(1200, 900, "Demo", None, None)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            # 2. Initialize visualization data structures
            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.opt)
            self.scene = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

            # 3. Install GLFW mouse and keyboard callbacks
            self.button_left = False
            self.button_middle = False
            self.button_right = False
            self.cursor_lastx = 0
            self.cursor_lasty = 0
            glfw.set_key_callback(self.window, self.keyboardCB)
            glfw.set_cursor_pos_callback(self.window, self.cursorPosCB)
            glfw.set_mouse_button_callback(self.window, self.mouseButtonCB)
            glfw.set_scroll_callback(self.window, self.scrollCB)

        if self.is_savedata:
            self.loop_data = 0
            self.data_freq = 50   # timestep(2ms) * 50 = 100ms
            self.csv_path = "demo.csv"
            with open(self.csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(["time", "q1", "q2"])

        # 5. init Controller
        self.initController()

    def initController(self):
        # 1. set camera & opt PRM
        if self.is_show:
            self.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
            self.cam.lookat = [0.012768, -0.000000, 1.254336]
            self.cam.distance = 6.0
            self.cam.azimuth = 90
            self.cam.elevation = -5
        # 2. set init pos
        self.data.qpos[0] = np.deg2rad(10)
        # 3. set the controller
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
        data.qfrc_applied = ctrl1
        # 2. PD + Feedforward
        # data.qfrc_applied = ctrl1 + ctrl2
        # 3. Feedback + Linearization
        # data.qfrc_applied = ctrl3 + ctrl2

        if self.is_savedata:
            self.saveData()

    def saveData(self):
        self.loop_data = self.loop_data + 1
        if self.loop_data % self.data_freq != 0:
            return
        with open(self.csv_path, "a+", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([self.data.time, self.data.qpos[0], self.data.qpos[1]])
        # print(self.data.time)
    def mainFun(self):
        simend = 100.0
        while True:
            simstart = self.data.time
            while (self.data.time - simstart) < 1.0 / 60.0:
                mj.mj_step(self.model, self.data)

            if self.data.time > simend:
                break

            if self.is_show and not glfw.window_should_close(self.window):
                # get framebuffer viewport
                viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
                viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

                # Update scene and render
                # self.cam.lookat[0] = self.data.qpos[0]
                mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                   mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                mj.mjr_render(viewport, self.scene, self.context)

                # swap OpenGL buffers (blocking call due to v-sync)
                glfw.swap_buffers(self.window)

                # process pending GUI events, call GLFW callbacks
                glfw.poll_events()

        if self.is_show and not glfw.window_should_close(self.window):
            glfw.terminate()

    # keyboard mode
    def keyboardCB(self, window, key, scancode, action, mods):
        if action == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            self.initController()

    def cursorPosCB(self, window, cursor_xpos, cursor_ypos):
        # compute mouse displacement, save
        cursor_dx = cursor_xpos - self.cursor_lastx
        cursor_dy = cursor_ypos - self.cursor_lasty
        self.cursor_lastx = cursor_xpos
        self.cursor_lasty = cursor_ypos

        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        elif self.button_middle:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, cursor_dx / height, cursor_dy / height, self.scene, self.cam)

    def mouseButtonCB(self, window, buttton, action, mods):
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        self.cursor_lastx, self.cursor_lasty = glfw.get_cursor_pos(window)

    def scrollCB(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)



if __name__ == "__main__":
    is_show = True
    is_savedata = True
    rel_path = "db_pendulum.xml"
    dirname = os.path.dirname(__file__)
    xml_path = os.path.join(dirname + "/" + rel_path)


    pendulumControl = DBPendulumControl(xml_path, is_show, is_savedata)
    pendulumControl.mainFun()
