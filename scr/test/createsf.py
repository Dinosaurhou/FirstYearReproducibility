import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def generate_sf_network(N, avg_k, gamma):
    """
    使用配置模型生成一个无标度网络 (Scale-Free Network)。

    该函数通过从一个带有指数gamma的离散幂律分布中抽样来生成度序列，
    然后使用该度序列构建一个网络。该方法可以确保节点数严格为N，
    并允许自定义平均度和幂律指数。

    Args:
        N (int): 网络中的节点数量 (Number of nodes)。
        avg_k (float): 期望的平均度 (Target average degree)。
        gamma (float): 幂律分布的指数 (Power-law exponent, gamma > 1)。

    Returns:
        networkx.Graph: 生成的无标度网络。如果无法生成有效的度序列，则返回 None。
    """
    # 1. 生成一个符合幂律分布的度序列
    # 为了从离散幂律分布中抽样，我们使用numpy的random.zipf函数。
    # Zipf分布的指数 a 和 幂律指数 gamma 的关系是 a = gamma。
    # 为了控制平均度，我们需要确定度序列的最小值 k_min。
    # 理论上，<k> = k_min * (gamma - 1) / (gamma - 2)
    # 因此，k_min = <k> * (gamma - 2) / (gamma - 1)
    # 我们将 k_min 取整，并确保其至少为1。
    if gamma <= 2:
        # 当 gamma <= 2 时，平均度会发散，上述公式不适用。
        # 在这种情况下，我们凭经验设置一个小的 k_min。
        k_min = 1
        print(f"Warning: gamma ({gamma}) <= 2, the theoretical average degree diverges. "
              f"Setting k_min empirically to {k_min}.")
    else:
        k_min = round(avg_k * (gamma - 2) / (gamma - 1))
        k_min = max(1, int(k_min)) # 确保 k_min >= 1

    # 使用Zeta分布（Zipf的推广）生成度序列
    # a 参数对应于 gamma
    s = np.random.zipf(a=gamma, size=N)
    
    # 将生成的样本映射到我们期望的度范围 [k_min, N-1]
    # 这是一个简单的线性变换，也可以使用更复杂的映射方法
    degrees = s + k_min - 1

    # 2. 调整度序列以满足图存在性的条件
    # a) 度的总和必须是偶数。如果不是，随机选择一个节点使其度+1。
    if sum(degrees) % 2 != 0:
        idx = np.random.randint(0, N)
        degrees[idx] += 1

    # b) 任何节点的度都不能超过 N-1。
    degrees[degrees > N - 1] = N - 1
    # 再次确保总度数为偶数
    if sum(degrees) % 2 != 0:
        idx = np.random.randint(0, N)
        degrees[idx] = min(N - 1, degrees[idx] + 1) if degrees[idx] < N - 1 else degrees[idx] - 1


    # 3. 使用配置模型创建图
    # configuration_model会根据给定的度序列生成一个（可能含有重边和自环的）图
    try:
        G = nx.configuration_model(degrees)
    except nx.NetworkXError as e:
        print(f"Error creating graph with configuration model: {e}")
        print("This might happen if the degree sequence is not graphically realizable.")
        return None

    # 4. 移除自环和重边，得到一个简单图
    G = nx.Graph(G)  # 移除重边
    G.remove_edges_from(nx.selfloop_edges(G)) # 移除自环

    print("--- Network Generation Summary ---")
    print(f"Target Parameters: N={N}, avg_k={avg_k}, gamma={gamma}")
    print(f"Generated Network Properties:")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    actual_avg_k = 2 * G.number_of_edges() / G.number_of_nodes()
    print(f"  - Actual Average Degree: {actual_avg_k:.4f}")
    print(f"  - k_min used for generation: {k_min}")
    print("---------------------------------")
    
    return G

def plot_degree_distribution(G):
    """
    计算并绘制网络的度分布图（双对数坐标）。
    """
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_counts = Counter(degree_sequence)
    k, Pk = zip(*degree_counts.items())

    plt.figure(figsize=(8, 6))
    plt.loglog(k, Pk, 'bo', markersize=5)
    plt.title("Degree Distribution (log-log scale)")
    plt.xlabel("Degree (k)")
    plt.ylabel("Count (P(k))")
    plt.grid(True, which="both", ls="--")
    plt.show()


if __name__ == '__main__':
    # --- 使用示例 ---
    # 参数设置
    N = 3000       # 节点数
    avg_k = 4      # 期望平均度
    gamma = 2.5    # 幂律指数

    # 生成网络
    sf_network = generate_sf_network(N, avg_k, gamma)

    if sf_network:
        # 验证网络基本属性
        print(f"Is the graph connected? {nx.is_connected(sf_network)}")
        
        # 绘制度分布图来验证其无标度特性
        plot_degree_distribution(sf_network)
