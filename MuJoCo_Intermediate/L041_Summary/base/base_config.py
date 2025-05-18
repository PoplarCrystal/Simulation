import numpy as np

class ConfigBase:
    class Sim:
        xml_scene_filename = "models/franka_emika_panda/scene.xml"
        dt = 0.001
        sim_time = 30
        sim_mode = "kin"  # "dyn", 选择是运动学仿真还是动力学仿真 

    class Render:
        is_render = True        # 是否打开渲染
        render_fps = 30         # 每step 10步，更新一次渲染
        show_left_ui = False    # 是否打开左右UI界面
        show_right_ui = False   # 是否打开左右UI界面
        cam_para = [0.012768, -0.000000, 1.254336, 10, 90, -5]  # 相机参数，lookat, distance, azimuth, elevation

