import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 可视化参数
N = 500  # 节点数，便于观察
AVG_DEGREE = 4
GAMMAS = [3.0, 2.7, 2.3]

# 生成SF网络的函数

def create_sf_graph(n, gamma, k_avg):
    attempts = 0
    while True:
        attempts += 1
        degrees = nx.utils.powerlaw_sequence(n, gamma)
        degrees = [max(1, int(d)) for d in degrees] # 确保最小度为1
        
        # 调整度序列以匹配平均度
        current_sum = sum(degrees)
        target_sum = n * k_avg
        if current_sum > 0:
            scale = target_sum / current_sum
            degrees = [max(1, int(d * scale)) for d in degrees]

        if sum(degrees) % 2 != 0:
            idx = np.random.randint(0, n)
            degrees[idx] += 1
        
        # 检查平均度是否在可接受范围内
        current_k_avg = sum(degrees) / n
        if abs(current_k_avg - k_avg) < 0.5 or attempts > 200:
            break

    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # 打印网络信息
    actual_avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    print(f"--- SF网络 (γ={gamma}) 生成信息 ---")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  实际平均度: {actual_avg_degree:.4f}")
    print(f"  目标平均度: {k_avg}")
    print("---------------------------------")
    
    return G

fig = plt.figure(figsize=(18, 6))
for i, gamma in enumerate(GAMMAS):
    if gamma == 3.0:
        m = int(AVG_DEGREE / 2)
        G = nx.barabasi_albert_graph(N, m)
        # 打印BA网络信息
        actual_avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
        print(f"--- BA网络 (γ≈3.0) 生成信息 ---")
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  边数: {G.number_of_edges()}")
        print(f"  实际平均度: {actual_avg_degree:.4f}")
        print("---------------------------------")
    else:
        G = create_sf_graph(N, gamma, AVG_DEGREE)
    # spring布局，3D坐标
    pos = nx.spring_layout(G, dim=3, seed=42)
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    zs = [pos[n][2] for n in G.nodes()]
    degrees = np.array([G.degree(n) for n in G.nodes()])
    # 节点颜色和大小反映度数
    node_colors = degrees
    node_sizes = 20 + 80 * (degrees - degrees.min()) / (degrees.max() - degrees.min())
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    # 画连边
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color='gray', alpha=0.3, linewidth=0.5)
    # 画节点
    ax.scatter(xs, ys, zs, c=node_colors, cmap='plasma', s=node_sizes, alpha=0.8)
    ax.set_title(f"SF网络 γ={gamma}")
    ax.set_axis_off()

plt.suptitle("不同幂律分布的无标度网络3D结构可视化 (N=1000, <k>=4)", fontsize=16)
plt.tight_layout()
plt.show()
