import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import time

model = mj.MjModel.from_xml_path("model/mjcf/scene.xml")
data = mj.MjData(model)

viewer = mjv.launch_passive(model, data)
viewer.opt.frame = mj.mjtFrame.mjFRAME_WORLD
viewer.cam.lookat = [0.012768, -0.000000, 1.254336]
viewer.cam.distance = 10
viewer.cam.azimuth = 90
viewer.cam.elevation = -5
sim_start, sim_end = time.time(), 100


# 打印所有关节信息
def print_joint_info(model, data):
    joint_type_enum = ["Free", "Ball", "Slide", "Hinge"]
    print("----------- Print Joint Information ---------------")
    print("Number of joints:", model.njnt)
    for i in range(model.njnt):
        print(f"Joint {i}:")
        print(f"  Name: {mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)}")
        print(f"  Type: {joint_type_enum[model.jnt_type[i]]}")  # 关节类型（0=自由,1=球,2=滑动,3=铰链）
        print(f"  Parent body name: {model.body(model.jnt_bodyid[i]).name}")

def print_motor_info(model, data):
    print("----------- Print Motor Information ---------------")
    for i in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"Actuator {i}: {name}")

def control(model, data):
    key_name = "home"
    pos_des = model.key("home").qpos
    vel_des = np.zeros_like(pos_des)
    kp = np.array([20, 200, 200, 200, 20, 20])
    kd = np.array([2, 20, 20, 20, 2, 2])
    data.ctrl = kp * (pos_des - data.qpos) + kd * (vel_des - data.qvel)

        
print_joint_info(model, data)
print_motor_info(model, data)
# 使用真实时间，人为地降低模拟循环的速度，使得模拟过程更易于观察和分析
while time.time() - sim_start < sim_end:
    step_start = time.time()
    render_loop, render_count = 10, 0
    # 1. 连续循环N个周期
    while render_count < render_loop:
        render_count += 1
        control(model, data)
        mj.mj_step(model, data)
    # 2. 开始render一次
    if viewer.is_running():  # exit
        viewer.sync()
    else:
        break
    # 3. 休眠到下一个周期
    step_next_delta = model.opt.timestep * render_loop - (time.time() - step_start)
    if step_next_delta > 0:
        time.sleep(step_next_delta)
# 4. 仿真结束，关闭render
viewer.close()