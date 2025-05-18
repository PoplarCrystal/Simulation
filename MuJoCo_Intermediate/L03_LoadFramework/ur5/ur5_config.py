import path
from base.base_config import ConfigBase
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(os.path.dirname(current_directory))
robot_name = "ur5"
model_directory =  os.path.join(parent_directory, "L00_Models/model", robot_name)

class ConfigUR5(ConfigBase):
    class Sim(ConfigBase.Sim):
        xml_scene_filename = model_directory + "/mjcf/scene.xml"
        sim_time = 50
        sim_mode = "dyn"  # "dyn", 选择是运动学仿真还是动力学仿真 

    class Render(ConfigBase.Render):
        show_left_ui = True    # 是否打开左右UI界面
        show_right_ui = False   # 是否打开左右UI界面


