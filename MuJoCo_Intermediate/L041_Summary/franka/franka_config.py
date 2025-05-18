
import path
from base.base_config import ConfigBase
MUJOCO_DIR = path.project_root


class ConfigFranka(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = MUJOCO_DIR + "/models/franka_emika_panda/scene.xml"
        sim_time = 300
        sim_mode = "dyn"  # "dyn", 选择是运动学仿真还是动力学仿真 

    class Render(ConfigBase.Render):
        show_left_ui = True    # 是否打开左右UI界面
        show_right_ui = False   # 是否打开左右UI界面
        cam_para = [0.20230409, 0.05751299, 1.0197645, 1.64, 90, -90]  # 相机参数，lookat, distance, azimuth, elevation


