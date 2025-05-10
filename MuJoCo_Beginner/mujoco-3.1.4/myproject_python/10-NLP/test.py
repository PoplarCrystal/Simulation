import numpy as np
# from scipy.spatial import distance
import nlopt

# 定义目标函数
def objective_function(x, grad):
    if len(grad):
        # 如果梯度被请求，则返回None，因为我们的目标函数不计算梯度
        return None
    return x[0]**2 + x[1]**2

# 创建NLopt算法实例
algorithm = nlopt.LN_COBYLA  # 使用COBYLA算法（无导数局部优化算法）
n = 2  # 变量维度
opt = nlopt.opt(algorithm, n)

# 设置目标函数
opt.set_min_objective(objective_function)

# 设置变量的上下界
lower_bounds = [-1.0, -1.0]
upper_bounds = [1.0, 1.0]
opt.set_lower_bounds(lower_bounds)
opt.set_upper_bounds(upper_bounds)

opt.set_xtol_rel(1e-5)

# 初始猜测值
initial_guess = [0.5, 0.5]

# 运行优化
result = opt.optimize(initial_guess)

# 获取最优解
optimal_solution = result

# 获取最优解处的目标函数值
min_value = opt.last_optimum_value()

print("Optimal solution:", optimal_solution)
print("Minimum value of the objective function:", min_value)