import nlopt
import numpy as np


# 最小化目标函数
def myCost(x, grad):
    cost = 0
    # cost = a1*x1^2 + a2*x2^2 + a3*x3^2 + a4*x4^2 + a5*x5^2
    para = 1
    cost = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2
    return cost


# 等式约束
def myEqualityContraint1(x, grad):
    # x1 + x2 + x3 = c1
    # x3*x3 + x4 = c2
    result = x[0] + x[1] + x[2] - 5

    return result

def myEqualityContraint2(x, grad):
    # x1 + x2 + x3 = c1
    # x3*x3 + x4 = c2
    result = x[2] * x[2] + x[3] - 2
    return result
    

# 不等式约束
def myInequalityContraints(x, grad):
    result = 0
    # x4 * x4 + x5 * x5 < c1
    x4 = x[3]
    x5 = x[4]
    result = x[3] * x[3] + x[4] * x[4] - 5.0
    return result



# 1. 选择优化算法
opt = nlopt.opt(nlopt.LN_COBYLA, 5)

# 2. 设置最小化目标函数
para1 = np.ones(5)
opt.set_min_objective(myCost)

# 3. 设置等于约束
para2 = np.array([5, 2])
# opt.add_equality_mconstraint(myEqualityContraints, [1e-8, 1e-8])
opt.add_equality_constraint(myEqualityContraint1, 1e-8)
opt.add_equality_constraint(myEqualityContraint2, 1e-8)

# 4. 设置不等于约束
para3 = np.array([5])
opt.add_inequality_constraint(myInequalityContraints, 1e-8)

# 5. 设置决策变量边界条件
lb = [0.3, -float('inf'), -float('inf'), -float('inf'), -float('inf')]
ub = [float('inf'), float('inf'), 5, float('inf'), float('inf')]
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)

# 6. 设置收敛阈值
opt.set_xtol_rel(1e-4)

# 7. 设置初始值
firstguess = [1.0, 1.0, 1., 2., 1.]

# 8. 求解优化问题
x = opt.optimize(firstguess)

# 打印求解信息
print("\noptimum at x1: %s, x2: %s, x3: %s, x4: %s, x5: %s," % (x[0], x[1], x[2], x[3], x[4]))
print("result code = ", opt.last_optimize_result())
