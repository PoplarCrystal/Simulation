import mujoco as mj
# from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np
import math
import nlopt
import time
import os
import csv


class BallControl:
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
            self.viewer.cam.lookat = [0.0, 0.0, 1.5]
            self.viewer.cam.distance = 5.0
            self.viewer.cam.azimuth = 89.608063
            self.viewer.cam.elevation = -11.588379
        # 2. init Controller
        self.init_controller()
        # 3. save data
        if self.is_save_data:
            self.data_count = 0
            self.data_freq = 50  # timestep(2ms) * 50 = 100ms
            self.csv_path = "demo.csv"
            with open(self.csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(["time", "q1", "q2"])

    def init_controller(self):
        self.target_pos = [5.0, 2.1]
        self.init_param = np.array([10.0, np.pi / 4, 2.0])
        self.nlp_solution()
        # 2. set the controller
        mj.set_mjcb_control(self.controller)

    def nlp_solution(self):
        # Define optimization problem
        opt = nlopt.opt(nlopt.LN_COBYLA, 3)
        # Define lower and upper bounds
        opt.set_lower_bounds([0.1, 0.1, 0.1])
        opt.set_upper_bounds([10000.0, np.pi / 2 - 0.1, 10000.0])
        # Set objective funtion
        opt.set_min_objective(self.cost_func)
        # Define equality constraints
        tol = [1e-4, 1e-4]
        opt.add_equality_mconstraint(self.equality_constraints, tol)
        # Set relative tolerance on optimization parameters
        opt.set_xtol_rel(1e-4)
        # Set the init value and solve problem
        sol = opt.optimize(self.init_param)
        print("------------------------")
        print("soulution: v, theta, tof: ", sol[0], sol[1], sol[2])
        # set the value
        v_sol, theta_sol = sol[0], sol[1]
        self.data.qvel[0] = v_sol * math.cos(theta_sol)
        self.data.qvel[2] = v_sol * math.sin(theta_sol)

    def cost_func(self, x, grad):
        cost = 0.0
        return cost

    def equality_constraints(self, result, x, grad):
        pos = self.simulator(x)
        result[0] = pos[0] - self.target_pos[0]
        result[1] = pos[1] - self.target_pos[1]

    def simulator(self, inputs):
        v, theta, tof = inputs[0], inputs[1], inputs[2]
        self.data.qvel[0] = v * math.cos(theta)
        self.data.qvel[2] = v * math.sin(theta)
        while self.data.time < tof:
            mj.mj_step(self.model, self.data)
        outputs = np.array([self.data.qpos[0], self.data.qpos[2]])
        mj.mj_resetData(self.model, self.data)
        return outputs

    def controller(self, model, data):
        pass

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
    is_show = True
    is_save_data = False

    ballControl = BallControl(xml_path, is_show, is_save_data)
    ballControl.main()
