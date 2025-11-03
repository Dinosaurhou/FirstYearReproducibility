import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 设置参数
N = 10000  # 节点数
avg_degree = 4  # 平均度
gamma = 2.7  # 幂指数

def generate_sf_network(N, gamma, target_avg_degree):
    """
    生成具有指定幂指数和平均度的无标度网络
    """
    # 计算目标总度数
    target_sum = N * target_avg_degree
    
    # 生成幂律度序列
    # 使用幂律分布: P(k) ~ k^(-gamma)
    # 度的最小值设为1，最大值设为N-1
    k_min = 1
    k_max = min(N - 1, 1000)  # 限制最大度避免不合理的值
    
    # 生成度序列
    degree_sequence = []
    for _ in range(N):
        # 使用逆变换采样生成幂律分布的度值
        u = np.random.random()
        if gamma != 1:
            k = k_min * ((k_max/k_min)**(1-gamma) - u * ((k_max/k_min)**(1-gamma) - 1))**(1/(1-gamma))
        else:
            k = k_min * (k_max/k_min)**u
        degree_sequence.append(int(k))
    
    # 调整度序列使总度数等于目标值
    current_sum = sum(degree_sequence)
    scale_factor = target_sum / current_sum
    
    degree_sequence = [max(1, int(d * scale_factor)) for d in degree_sequence]
    
    # 微调：确保总度数恰好等于目标值且为偶数
    current_sum = sum(degree_sequence)
    diff = int(target_sum - current_sum)
    
    # 确保总度数为偶数
    if int(target_sum) % 2 != 0:
        diff += 1
    
    # 分配差值到度序列中（优先分配给度较大的节点）
    sorted_indices = np.argsort(degree_sequence)[::-1]
    i = 0
    while diff != 0:
        idx = sorted_indices[i % len(sorted_indices)]
        if diff > 0:
            degree_sequence[idx] += 1
            diff -= 1
        elif diff < 0 and degree_sequence[idx] > 1:
            degree_sequence[idx] -= 1
            diff += 1
        i += 1
        if i > len(sorted_indices) * 2:  # 防止无限循环
            break
    
    # 确保总度数为偶数
    if sum(degree_sequence) % 2 != 0:
        degree_sequence[sorted_indices[0]] += 1
    
    # 使用配置模型生成网络
    try:
        G = nx.configuration_model(degree_sequence)
        G = nx.Graph(G)  # 转换为简单图，移除重边和自环
        G.remove_edges_from(nx.selfloop_edges(G))
    except:
        print("配置模型生成失败，尝试使用expected_degree_graph")
        G = nx.expected_degree_graph(degree_sequence, selfloops=False)
    
    return G

# 生成网络
print("正在生成网络...")
G = generate_sf_network(N, gamma, avg_degree)

# 打印网络统计信息
print(f"\n=== 网络统计信息 ===")
print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")
actual_avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
print(f"实际平均度: {actual_avg_degree:.4f}")
print(f"目标平均度: {avg_degree}")
print(f"误差: {abs(actual_avg_degree - avg_degree):.4f}")

# 分析度分布
degrees = [G.degree(n) for n in G.nodes()]
print(f"\n最小度: {min(degrees)}")
print(f"最大度: {max(degrees)}")
print(f"中位数度: {np.median(degrees):.2f}")

# 验证幂律指数
from scipy.stats import linregress
degree_counts = np.bincount(degrees)
degrees_unique = np.nonzero(degree_counts)[0]
counts = degree_counts[degrees_unique]

# 对度大于等于2的部分进行线性拟合（对数坐标）
mask = degrees_unique >= 2
log_k = np.log(degrees_unique[mask])
log_pk = np.log(counts[mask])
slope, intercept, r_value, _, _ = linregress(log_k, log_pk)
estimated_gamma = -slope
print(f"\n拟合的幂指数 γ: {estimated_gamma:.2f}")
print(f"目标幂指数 γ: {gamma}")
print(f"R²: {r_value**2:.4f}")

# 可视化度分布
plt.figure(figsize=(12, 5))

# 子图1：对数坐标的度分布
plt.subplot(1, 2, 1)
plt.loglog(degrees_unique, counts, 'bo', markersize=4, alpha=0.6, label='实际数据')
plt.loglog(degrees_unique[mask], np.exp(intercept + slope * log_k), 'r-', 
           linewidth=2, label=f'拟合: γ={estimated_gamma:.2f}')
plt.xlabel('度 k', fontsize=12)
plt.ylabel('频数 P(k)', fontsize=12)
plt.title(f'度分布 (对数坐标)\nN={N}, 平均度={actual_avg_degree:.2f}', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：度分布直方图
plt.subplot(1, 2, 2)
plt.hist(degrees, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('度 k', fontsize=12)
plt.ylabel('节点数', fontsize=12)
plt.title('度分布直方图', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 保存网络（可选）
# nx.write_edgelist(G, "sf_network.txt")