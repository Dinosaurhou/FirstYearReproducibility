import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d

# 设置中文字体支持（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义 x 的范围
x = np.linspace(0.01, 2, 2000)  # 从0.01开始避免0点的问题，增加点数提高精度

# 第一个函数: y = x
y1 = x

# 第二个函数: y = e^(-((1-x^6)(1-x^12))/(6*x^6+12*x^12-18*x^18))
# 为了避免除零和数值错误，需要特殊处理
def calculate_y2(x_input):
    x_arr = np.atleast_1d(x_input)
    x6 = x_arr**6
    x12 = x_arr**12
    x18 = x_arr**18
    
    numerator = (1 - x6) * (1 - x12)
    denominator = 6*x6 + 12*x12 - 18*x18
    
    # 处理分母为零的情况
    y2 = np.zeros_like(x_arr, dtype=float)
    
    # 只在分母不为零的地方计算
    mask = np.abs(denominator) > 1e-10
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        exponent = np.where(mask, -numerator / denominator, 0)
        # 限制指数范围，避免溢出
        exponent = np.clip(exponent, -100, 100)
        y2 = np.where(mask, np.exp(exponent), np.nan)
    
    return y2 if isinstance(x_input, np.ndarray) else y2[0]

# 定义 pc 计算函数
def calculate_pc(r):
    """
    计算 pc = 1 / (6*r^6 + 12*r^12 - 18*r^18)
    其中 r 是交点的 y 值
    """
    r6 = r**6
    r12 = r**12
    r18 = r**18
    
    denominator = 6*r6 + 12*r12 - 18*r18
    
    if abs(denominator) < 1e-10:
        return np.nan
    
    pc = 1.0 / denominator
    return pc

y2 = calculate_y2(x)

# 方法1: 直接查找符号变化
def find_intersections_sign_change(x_vals, y1_vals, y2_vals):
    """通过查找差值函数的符号变化来找交点"""
    intersections = []
    
    # 计算差值
    diff = y1_vals - y2_vals
    
    # 找到符号变化的位置
    for i in range(len(diff) - 1):
        if np.isnan(diff[i]) or np.isnan(diff[i+1]):
            continue
        
        # 检查符号是否改变
        if diff[i] * diff[i+1] < 0:
            # 使用线性插值估计交点位置
            x_intersect = x_vals[i] - diff[i] * (x_vals[i+1] - x_vals[i]) / (diff[i+1] - diff[i])
            y_intersect = x_intersect  # 因为第一个函数是 y = x
            
            # 验证
            y2_check = calculate_y2(x_intersect)
            if not np.isnan(y2_check) and abs(y_intersect - y2_check) < 0.01:
                intersections.append((x_intersect, y_intersect))
    
    return intersections

# 方法2: 使用 brentq 方法（更稳健）
def find_intersections_brentq(x_vals, y1_vals, y2_vals):
    """使用 brentq 方法寻找交点"""
    intersections = []
    
    def difference_func(x_val):
        y1_val = x_val
        y2_val = calculate_y2(x_val)
        if np.isnan(y2_val):
            return np.inf
        return y1_val - y2_val
    
    # 计算差值
    diff = y1_vals - y2_vals
    
    # 找到符号变化的区间
    for i in range(len(diff) - 1):
        if np.isnan(diff[i]) or np.isnan(diff[i+1]):
            continue
        
        # 检查符号是否改变
        if diff[i] * diff[i+1] < 0:
            try:
                # 使用 brentq 在这个区间内精确求解
                x_intersect = brentq(difference_func, x_vals[i], x_vals[i+1])
                y_intersect = x_intersect
                
                # 验证
                y2_check = calculate_y2(x_intersect)
                if not np.isnan(y2_check) and abs(y_intersect - y2_check) < 1e-6:
                    # 检查是否重复
                    is_duplicate = False
                    for existing_x, existing_y in intersections:
                        if abs(x_intersect - existing_x) < 0.001:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        intersections.append((x_intersect, y_intersect))
            except:
                pass
    
    return intersections

# 使用两种方法寻找交点
intersections = find_intersections_brentq(x, y1, y2)

# 如果方法2没找到，尝试方法1
if not intersections:
    intersections = find_intersections_sign_change(x, y1, y2)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制两个函数
plt.plot(x, y1, label='y = x', linewidth=2, color='blue')
plt.plot(x, y2, label=r'$y = e^{-\frac{(1-x^6)(1-x^{12})}{6x^6+12x^{12}-18x^{18}}}$', 
         linewidth=2, color='red')

# 标记交点并计算 pc
if intersections:
    print(f"\n{'='*70}")
    print(f"找到 {len(intersections)} 个交点:")
    print(f"{'='*70}")
    
    for i, (x_int, y_int) in enumerate(intersections):
        # 计算 pc 值
        pc_value = calculate_pc(y_int)
        
        print(f"\n交点 {i+1}:")
        print(f"  x = {x_int:.8f}")
        print(f"  y = {y_int:.8f}")
        print(f"  pc = 1/(6*r^6 + 12*r^12 - 18*r^18) = {pc_value:.8f}")
        
        # 在图上标记交点
        plt.plot(x_int, y_int, 'go', markersize=12, markeredgecolor='black', 
                markeredgewidth=2, label='交点' if i == 0 else '', zorder=5)
        
        # 添加文本注释（包含 pc 值）
        annotation_text = f'({x_int:.4f}, {y_int:.4f})\npc = {pc_value:.4f}'
        plt.annotate(annotation_text, 
                    xy=(x_int, y_int), 
                    xytext=(15, 15), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                  lw=1.5, color='black'))
    
    print(f"\n{'='*70}")
else:
    print("未找到有效交点")

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

# 添加坐标轴
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axvline(x=0, color='black', linewidth=0.5)

# 设置标签和标题
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.title('函数图像对比（含交点标记及pc值）', fontsize=15, fontweight='bold')

# 添加图例
plt.legend(fontsize=11, loc='best')

# 设置坐标轴范围
plt.xlim(0, 2)
plt.ylim(0, 2)

# 显示图形
plt.tight_layout()
plt.show()

# 如果需要保存图像
# plt.savefig('function_plots_with_intersections.png', dpi=300, bbox_inches='tight')