import nlopt
import numpy as np


# 最小化目标函数
def myCost(x, grad, para):
    cost = 0
    # cost = a1*x1^2 + a2*x2^2 + a3*x3^2 + a4*x4^2 + a5*x5^2
    cost = para[0] * x[0]**2 + para[1] * x[1]**2 + para[2] * x[2]**2 + para[3] * x[3]**2 + para[4] * x[4]**2
    return cost


# 等式约束
def myEqualityContraints(result, x, grad, para):
    # x1 + x2 + x3 = c1
    # x3*x3 + x4 = c2
    result[0] = x[0] + x[1] + x[2] - para[0]
    result[1] = x[2] * x[2] + x[3] - para[1]

# 不等式约束
def myInequalityContraints(x, grad, para):
    result = 0
    # x4 * x4 + x5 * x5 < c1
    result = x[3] * x[3] + x[4] * x[4] - para[0]
    return result



# 1. 选择优化算法
opt = nlopt.opt(nlopt.LN_COBYLA, 5)

# 2. 设置最小化目标函数
para1 = np.ones(5)
opt.set_min_objective(lambda x, grad: myCost(x, grad, para1))

# 3. 设置等于约束
para2 = np.array([5, 2])
opt.add_equality_mconstraint(lambda result, x, grad: myEqualityContraints(result, x, grad, para2), np.array([1e-8, 1e-8]))

# 4. 设置不等于约束
para3 = np.array([5])
opt.add_inequality_constraint(lambda x, grad: myInequalityContraints(x, grad, para3), 1e-8)

# 5. 设置决策变量边界条件
lb = np.array([0.3, -np.inf, -np.inf, -np.inf, -np.inf])
ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

# 6. 设置收敛阈值与最大迭代次数
opt.set_xtol_rel(1e-4)
opt.set_maxeval(10000)

# 7. 设置初始值
x0 = np.array([1, 1, 1, 2, 1])

# 8. 求解优化问题
x = opt.optimize(x0)

# 打印求解信息
print("\noptimum at x1: %f, x2: %f, x3: %f, x4: %f, x5: %f," % (x[0], x[1], x[2], x[3], x[4]))
print("min value =", opt.last_optimum_value())
print("result code =", opt.last_optimize_result())
