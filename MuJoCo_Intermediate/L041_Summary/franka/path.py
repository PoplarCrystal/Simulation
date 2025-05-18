import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # franka 目录
project_root = os.path.dirname(current_dir)  # project 目录
sys.path.append(project_root)  # 添加 project 到 Python 路径
