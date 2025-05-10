import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

class BallControl:
    def __init__(self, xml_path, is_show):
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.is_show = is_show

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

        # 5. init Controller
        self.initController()

    def initController(self):
        # 1. set camera PRM
        if self.is_show:
            self.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            self.cam.lookat = [0.0, 0.0, 0.0]
            self.cam.distance = 8.0
            self.cam.azimuth = 90
            self.cam.elevation = -45
        # 2. set init pos
        self.data.qpos[0] = 0.1
        self.data.qvel[0] = 2.0
        self.data.qvel[2] = 5.0
        # 3. set the controller
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
                self.cam.lookat[0] = self.data.qpos[0]
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
    rel_path = "ball.xml"
    dirname = os.path.dirname(__file__)
    xml_path = os.path.join(dirname + "/" + rel_path)


    ballControl = BallControl(xml_path, is_show)
    ballControl.mainFun()
