import numpy as np
import matplotlib.pyplot as plt



def critical_values_equal_degrees(a, b):
    """当a=b时的解析解"""
    f_c = 0.28467
    p_c = 2.4554 / a
    mu_inf =1.2564/a            #p_c * (1 - f_c) ** 2
    return a * p_c, a * mu_inf



def solve_for_ratio(ratio, b=4.0):
    """数值求解给定a/b比值时的临界值"""
    a = ratio * b


    def equation(r, a, b):
        return r - np.exp(-(1 - r ** a) * (1 - r ** b) / (a * r ** a + b * r ** b - (a + b) * r ** (a + b)))

    r_values = np.linspace(0.01, 0.99, 1000)
    eq_values = [equation(r, a, b) for r in r_values]

    idx = np.argmin(np.abs(eq_values))
    r_c = r_values[idx]

    p_c = 1 / (a * r_c ** a + b * r_c ** b - (a + b) * r_c ** (a + b))
    mu_inf = p_c * (1 - r_c ** a) * (1 - r_c ** b)

    return a * p_c, a * mu_inf


ratio_values = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

pc_scaled = []
mu_scaled = []

for ratio in ratio_values:
    if ratio == 1.0:

        p_val, mu_val = critical_values_equal_degrees(4.0, 4.0)
    else:
        p_val, mu_val = solve_for_ratio(ratio)
    pc_scaled.append(p_val)
    mu_scaled.append(mu_val)

plt.figure(figsize=(8, 6))

# 修改部分：去掉 'o-' 和 's-' 中的标记符号，只保留 '-' (实线)
# 蓝色曲线
plt.plot(ratio_values, pc_scaled, '-', linewidth=2,
         label=r'$\langle k \rangle_A p_c$', color='blue')

# 红色曲线
plt.plot(ratio_values, mu_scaled, '-', linewidth=2,
         label=r'$\langle k \rangle_A \mu_{\infty}(p_c)$', color='red')

# --- 修改开始 ---
# 1. 强制设置 x 和 y 的显示范围从 0 开始
plt.xlim(left=0, right=1.0)
plt.ylim(bottom=0, top=2.6)

# 2. 消除坐标轴与数据之间的留白 (可选，但推荐用于精确对齐)
plt.margins(x=0, y=0)

# 3. 设置 x 轴刻度 (只保留一位小数的刻度点)
x_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.xticks(x_ticks, [f'{x:.1f}' for x in x_ticks], fontsize=10)


plt.yticks(np.arange(0, 2.8, 0.2), fontsize=12)

plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.title('ER Networks: Critical Thresholds', fontsize=14)
plt.tight_layout()
plt.show()


#print("关键点数值:")
#key_ratios = [0.01, 0.05, 0.1, 0.5, 1.0]
#for ratio in key_ratios:
    #idx = np.where(ratio_values == ratio)[0][0]
   # print(f"a/b = {ratio:.2f}: ⟨k⟩_A p_c = {pc_scaled[idx]:.4f}, ⟨k⟩_A μ_∞ = {mu_scaled[idx]:.4f}")