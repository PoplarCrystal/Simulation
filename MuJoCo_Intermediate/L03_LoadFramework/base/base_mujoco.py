import mujoco as mj
import mujoco.viewer as mjv
import time
from .base_config import ConfigBase


class MuJoCoBase:
    def __init__(self, cfg : ConfigBase):
        # 1. 加载MuJoCo模型与数据
        self.model = mj.MjModel.from_xml_path(cfg.Sim.xml_scene_filename)
        self.data = mj.MjData(self.model)
        self._sim_time = cfg.Sim.sim_time
        self.model.opt.timestep = cfg.Sim.dt
        self._sim_mode = cfg.Sim.sim_mode
        # 2. 加载MuJoCo Scene
        self._is_render = cfg.Render.is_render
        self._render_fps = cfg.Render.render_fps
        if self._is_render:
            self.viewer = mjv.launch_passive(model=self.model, data=self.data, 
                                             show_left_ui=cfg.Render.show_left_ui,
                                             show_right_ui=cfg.Render.show_right_ui,
                                             key_callback=self.keyboard_cb)
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
            cam_para = cfg.Render.cam_para
            self.viewer.cam.lookat = cam_para[0:3]
            self.viewer.cam.distance = cam_para[3]
            self.viewer.cam.azimuth = cam_para[4]
            self.viewer.cam.elevation = cam_para[5]
        

    def simulation(self):
        sim_start = time.time()
        while time.time() - sim_start < self._sim_time:
            step_start = time.time()
            render_count = 0
            # 1. 连续循环N个周期
            while render_count < self._render_fps:
                render_count += 1
                self.pre_step()
                self.step()
                self.post_step()
            # 2. 开始render一次
            if self._is_render:
                if self.viewer.is_running():  # exit
                    self.viewer.sync()
                else:
                    break
            # 3. 休眠到下一个周期
                time_until_next_step = self.model.opt.timestep * self._render_fps - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        # 4. 仿真结束，关闭render
        if self._is_render:
            self.viewer.close()
        

    def pre_step(self):
        pass

    def step(self):
        if self._sim_mode == "kin":
            mj.mj_forward(self.model, self.data)
        elif self._sim_mode == "dyn":
            mj.mj_step(self.model, self.data)
            mj.mj_rnePostConstraint(self.model, self.data)

    def post_step(self):
        pass

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def keyboard_cb(self, keycode):
        pass


# if __name__ == "__main__":
#     cfg = ConfigBase()
#     ur5 = MuJoCoBase(cfg)
#     ur5.simulation()