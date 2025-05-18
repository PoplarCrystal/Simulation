import numpy as np
import os

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
robot_name = "unitree_g1"
model_directory =  os.path.join(parent_directory, "00-Models/model", robot_name)

class Conf:
    xml_scene_directory = model_directory + "/mjcf"
    xml_scene_filename = xml_scene_directory + "/scene.xml"
    xml_robot_directory = model_directory + "/mjcf"
    xml_robot_filename =xml_robot_directory + "/g1_12dof.xml"
    urdf_robot_directory = model_directory + "/urdf"
    urdf_robot_filename = urdf_robot_directory + "/g1_12dof.urdf"