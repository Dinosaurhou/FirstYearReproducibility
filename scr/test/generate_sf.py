import networkx as nx
import numpy as np
from typing import Tuple
import warnings


def generate_scale_free_network(N: int, avg_degree: float, gamma: float, 
                                max_attempts: int = 100, seed: int = None) -> nx.Graph:
    """
    生成无标度网络（Scale-Free Network）
    
    该方法使用配置模型（Configuration Model）结合幂律度分布来生成无标度网络。
    当 gamma=3 时，使用 Barabási-Albert (BA) 模型。
    
    参数:
    ----
    N : int
        网络节点数
    avg_degree : float
        期望平均度 <k>
    gamma : float
        幂律指数 λ (gamma)，范围 [2, 3]
    max_attempts : int, optional
        最大尝试次数，默认 100
    seed : int, optional
        随机种子，用于结果可重复性
        
    返回:
    ----
    G : networkx.Graph
        生成的无标度网络
        
    异常:
    ----
    ValueError
        当参数不符合要求时抛出
        
    参考文献:
    --------
    [1] Barabási, A. L., & Albert, R. (1999). Science, 286(5439), 509-512.
    [2] Newman, M. E. (2003). SIAM review, 45(2), 167-256.
    [3] Clauset, A., et al. (2009). SIAM review, 51(4), 661-703.
    
    示例:
    ----
    >>> G = generate_scale_free_network(N=1000, avg_degree=6.0, gamma=2.5, seed=42)
    >>> print(f"节点数: {G.number_of_nodes()}")
    >>> print(f"平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    """
    
    # 参数验证
    if N <= 0:
        raise ValueError("节点数 N 必须大于 0")
    if avg_degree <= 0:
        raise ValueError("平均度必须大于 0")
    if not (2 <= gamma <= 3):
        raise ValueError("幂律指数 gamma 必须在 [2, 3] 范围内")
    if avg_degree >= N:
        raise ValueError("平均度必须小于节点数")
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 特殊情况：gamma = 3 时使用 BA 模型
    if abs(gamma - 3.0) < 1e-6:
        return _generate_ba_network(N, avg_degree, seed)
    
    # 使用配置模型生成无标度网络
    # 尝试max_attempts次以满足平均度要求
    for attempt in range(max_attempts):
        try:
            G = _generate_sf_configuration_model(N, avg_degree, gamma, seed)
            
            # 验证生成的网络
            actual_avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            
            if abs(actual_avg_degree - avg_degree) <= 0.08:
                return G
                
        except Exception as e:
            if attempt == max_attempts - 1:
                warnings.warn(f"达到最大尝试次数，最后一次错误: {e}")
    
    raise RuntimeError(f"无法在 {max_attempts} 次尝试内生成符合要求的网络")


def _generate_ba_network(N: int, avg_degree: float, seed: int = None) -> nx.Graph:
    """
    使用 Barabási-Albert 模型生成网络
    
    BA 模型对应 gamma = 3 的情况
    """
    m = int(avg_degree / 2)
    if m < 1:
        m = 1
    
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    
    # 确保是简单图（无自环和多重边）
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G


def _generate_sf_configuration_model(N: int, avg_degree: float, gamma: float, 
                                     seed: int = None) -> nx.Graph:
    """
    使用配置模型生成无标度网络
    
    通过幂律度分布生成度序列，然后使用配置模型构建网络
    """
    # 计算度分布的最小度 k_min
    # 使用公式: <k> = (gamma-1)/(gamma-2) * k_min (对于 gamma < 3)
    k_min = max(1, int(avg_degree * (gamma - 2) / (gamma - 1)))
    
    # 生成度序列
    degree_sequence = _generate_power_law_degree_sequence(N, gamma, k_min, avg_degree)
    
    # 确保度序列之和为偶数（配置模型要求）
    if sum(degree_sequence) % 2 != 0:
        # 随机选择一个节点增加度数
        idx = np.random.randint(0, len(degree_sequence))
        if degree_sequence[idx] < N - 1:
            degree_sequence[idx] += 1
        else:
            degree_sequence[idx] -= 1
    
    # 使用配置模型生成网络
    G = nx.configuration_model(degree_sequence, seed=seed)
    
    # 转换为简单图（移除自环和多重边）
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # 移除孤立节点（如果有）
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # 如果节点数不足，添加节点并连接
    if G.number_of_nodes() < N:
        for i in range(G.number_of_nodes(), N):
            G.add_node(i)
            # 优先连接方式连接新节点
            if G.number_of_nodes() > 1:
                target = np.random.choice(list(G.nodes())[:-1])
                G.add_edge(i, target)
    
    return G


def _generate_power_law_degree_sequence(N: int, gamma: float, k_min: int, 
                                       target_avg: float) -> list:
    """
    生成符合幂律分布的度序列
    
    使用逆变换采样方法从幂律分布中采样
    """
    # 设置最大度以避免hub节点过大
    k_max = min(int(N ** 0.5), int(target_avg * 10))
    
    # 计算归一化常数
    normalization = sum([k ** (-gamma) for k in range(k_min, k_max + 1)])
    
    degree_sequence = []
    current_sum = 0
    
    for _ in range(N):
        # 从幂律分布中采样
        r = np.random.random()
        cumsum = 0
        
        for k in range(k_min, k_max + 1):
            cumsum += (k ** (-gamma)) / normalization
            if r <= cumsum:
                degree_sequence.append(k)
                current_sum += k
                break
        else:
            degree_sequence.append(k_min)
            current_sum += k_min
    
    # 调整度序列使平均度接近目标值
    current_avg = current_sum / N
    
    if abs(current_avg - target_avg) > 0.1:
        # 需要调整
        adjustment_factor = target_avg / current_avg
        degree_sequence = [max(k_min, int(d * adjustment_factor)) for d in degree_sequence]
    
    return degree_sequence


def analyze_network(G: nx.Graph) -> dict:
    """
    分析网络的基本统计特性
    
    返回:
    ----
    stats : dict
        包含网络统计信息的字典
    """
    degrees = [d for n, d in G.degree()]
    
    stats = {
        'N': G.number_of_nodes(),
        'E': G.number_of_edges(),
        'avg_degree': np.mean(degrees),
        'std_degree': np.std(degrees),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'clustering': nx.average_clustering(G),
        'connected_components': nx.number_connected_components(G)
    }
    
    return stats


# 示例使用
if __name__ == "__main__":
    # 测试不同的 gamma 值
    test_cases = [
        (1000, 6.0, 2.1),
        (1000, 6.0, 2.5),
        (1000, 6.0, 2.8),
        (1000, 6.0, 3.0),  # BA 模型
    ]
    
    for N, avg_k, gamma in test_cases:
        print(f"\n{'='*60}")
        print(f"生成网络: N={N}, <k>={avg_k}, γ={gamma}")
        print(f"{'='*60}")
        
        G = generate_scale_free_network(N, avg_k, gamma)
        stats = analyze_network(G)
        
        print(f"节点数: {stats['N']}")
        print(f"边数: {stats['E']}")
        print(f"平均度: {stats['avg_degree']:.4f} (目标: {avg_k})")
        print(f"度标准差: {stats['std_degree']:.4f}")
        print(f"最大度: {stats['max_degree']}")
        print(f"最小度: {stats['min_degree']}")
        print(f"聚类系数: {stats['clustering']:.4f}")
        print(f"连通分量数: {stats['connected_components']}")
        
        # 检查平均度误差
        error = abs(stats['avg_degree'] - avg_k)
        print(f"平均度误差: {error:.4f} {'✓' if error <= 0.08 else '✗'}")