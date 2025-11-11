import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

def get_giant_component(G):
    """返回图G的最大连通分量"""
    if not G.nodes():
        return nx.Graph()
    # 找到所有连通分量并按大小排序
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    # 最大连通分量是第一个
    giant = G.subgraph(components[0])
    return giant

def simulate_cascade(G_A, G_B, initial_attack_fraction):
    """
    模拟相互依赖网络的级联失效过程

    参数:
    - G_A, G_B: 两个networkx图对象，代表相互依赖的网络。
              我们假设节点 i in G_A 依赖于 节点 i in G_B。
    - initial_attack_fraction: 在网络A中初始移除的节点比例 (0到1之间)。

    返回:
    - 最终稳定后，网络A和B中剩余节点的比例。
    """
    # 复制原始网络以进行模拟，避免修改原图
    net_A = G_A.copy()
    net_B = G_B.copy()
    
    N = len(G_A.nodes())

    # --- 1. 初始攻击 ---
    # 随机选择要移除的节点
    nodes_to_remove_A = np.random.choice(
        list(net_A.nodes()), 
        size=int(N * initial_attack_fraction), 
        replace=False
    )
    net_A.remove_nodes_from(nodes_to_remove_A)
    
    # 记录每次迭代开始时节点的数量，用于判断是否达到稳定
    last_A_nodes_count = len(net_A.nodes())
    last_B_nodes_count = len(net_B.nodes())

    # --- 2. 级联失效循环 ---
    while True:
        # --- a. 网络内部失效 (A) ---
        # 节点必须属于最大连通分量才能发挥功能
        giant_A = get_giant_component(net_A)
        nodes_to_remove_A = set(net_A.nodes()) - set(giant_A.nodes())
        if nodes_to_remove_A:
            net_A.remove_nodes_from(nodes_to_remove_A)

        # --- b. 跨网络依赖失效 (A -> B) ---
        # A中失效的节点会导致B中对应的依赖节点失效
        failed_nodes_in_A = set(G_A.nodes()) - set(net_A.nodes())
        nodes_to_remove_B = set(net_B.nodes()) & failed_nodes_in_A
        if nodes_to_remove_B:
            net_B.remove_nodes_from(nodes_to_remove_B)

        # --- c. 网络内部失效 (B) ---
        giant_B = get_giant_component(net_B)
        nodes_to_remove_B = set(net_B.nodes()) - set(giant_B.nodes())
        if nodes_to_remove_B:
            net_B.remove_nodes_from(nodes_to_remove_B)

        # --- d. 跨网络依赖失效 (B -> A) ---
        failed_nodes_in_B = set(G_B.nodes()) - set(net_B.nodes())
        nodes_to_remove_A = set(net_A.nodes()) & failed_nodes_in_B
        if nodes_to_remove_A:
            net_A.remove_nodes_from(nodes_to_remove_A)
            
        # --- 3. 检查是否稳定 ---
        # 如果在一轮完整的迭代后没有节点被移除，则系统达到稳定
        current_A_nodes_count = len(net_A.nodes())
        current_B_nodes_count = len(net_B.nodes())
        
        if (current_A_nodes_count == last_A_nodes_count and
            current_B_nodes_count == last_B_nodes_count):
            break
        
        last_A_nodes_count = current_A_nodes_count
        last_B_nodes_count = current_B_nodes_count

    # 返回最终剩余节点的比例
    final_fraction = len(net_A.nodes()) / N
    return final_fraction

def create_sf_graph(n, gamma, k_avg):
    """
    使用配置模型创建具有特定幂律指数和平均度的无标度网络。
    """
    # 使用 `nx.utils.powerlaw_sequence` 生成度序列
    # 为确保平均度，我们需要调整 `powerlaw_sequence` 的 `tries` 参数
    # 并且可能需要多次尝试以获得接近的平均度
    
    # 循环直到生成的度序列满足要求
    while True:
        # 生成度序列
        degrees = nx.utils.powerlaw_sequence(n, gamma)
        # 将度序列转换为整数
        degrees = [int(d) for d in degrees]
        
        # 确保度序列之和为偶数，这是构建图的前提
        if sum(degrees) % 2 != 0:
            # 如果是奇数，随机选择一个节点使其度数+1
            idx = np.random.randint(0, n)
            degrees[idx] += 1
            
        # 检查平均度是否在合理范围内
        current_k_avg = sum(degrees) / n
        if k_avg - 0.5 < current_k_avg < k_avg + 0.5:
            break # 如果平均度接近目标值，则跳出循环

    # 使用配置模型创建图
    G = nx.configuration_model(degrees)
    # 移除自环和平行边，得到一个简单图
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


if __name__ == '__main__':
    # --- 网络参数 ---
    N = 50000
    AVG_DEGREE = 4

    # --- 模拟参数 ---
    attack_fractions = np.linspace(0, 1, 51)
    num_runs = 4 # 对于大网络，减少运行次数以节省时间

    # --- 定义要测试的网络类型 ---
    network_configs = {
        "ER": {"type": "ER"},
        "RR": {"type": "RR"},
        "SF (γ=3.0)": {"type": "SF", "gamma": 3.0},
        "SF (γ=2.7)": {"type": "SF", "gamma": 2.7},
        "SF (γ=2.3)": {"type": "SF", "gamma": 2.3},
    }

    # --- 准备绘图 ---
    plt.figure(figsize=(12, 8))
    
    # --- 对每种网络类型运行模拟 ---
    for name, config in network_configs.items():
        start_time = time.time()
        print(f"\n开始模拟 {name} 网络...")

        # --- 生成两个相互依赖的网络 ---
        print("  正在生成网络 A 和 B...")
        if config["type"] == "ER":
            P = AVG_DEGREE / (N - 1)
            G_A = nx.erdos_renyi_graph(N, P)
            G_B = nx.erdos_renyi_graph(N, P)
        elif config["type"] == "RR":
            # 随机正则图的度必须是整数
            d = int(AVG_DEGREE)
            G_A = nx.random_regular_graph(d, N)
            G_B = nx.random_regular_graph(d, N)
        elif config["type"] == "SF":
            gamma = config["gamma"]
            # 对于gamma=3.0，可以使用更高效的BA模型
            if gamma == 3.0:
                m = int(AVG_DEGREE / 2)
                G_A = nx.barabasi_albert_graph(N, m)
                G_B = nx.barabasi_albert_graph(N, m)
            else:
                # 对于其他gamma值，使用配置模型
                G_A = create_sf_graph(N, gamma, AVG_DEGREE)
                G_B = create_sf_graph(N, gamma, AVG_DEGREE)
        
        # 确保图是连通的，只取最大连通分量进行分析
        G_A = get_giant_component(G_A)
        G_B = get_giant_component(G_B)
        # 重新映射节点标签，使其从0到N-1，并保持一一对应
        G_A = nx.convert_node_labels_to_integers(G_A, first_label=0)
        # 创建一个与G_A节点匹配的G_B
        nodes_map_B = {old_label: new_label for new_label, old_label in enumerate(list(G_B.nodes())[:len(G_A.nodes())])}
        G_B = nx.relabel_nodes(G_B.subgraph(list(nodes_map_B.keys())), nodes_map_B)


        print(f"  网络生成完毕。网络A节点数: {len(G_A.nodes())}, 网络B节点数: {len(G_B.nodes())}")
        
        final_sizes = []
        for p_attack in attack_fractions:
            avg_size = 0
            for i in range(num_runs):
                final_size = simulate_cascade(G_A, G_B, p_attack)
                avg_size += final_size
            avg_size /= num_runs
            final_sizes.append(avg_size)
            print(f"  {name}, 初始攻击比例 p = {p_attack:.2f}, 最终网络规模比例 = {avg_size:.3f}")

        # --- 在图表中绘制当前网络类型的结果 ---
        plt.plot(attack_fractions, final_sizes, 'o-', label=name)
        
        end_time = time.time()
        print(f"  {name} 模拟完成，耗时: {end_time - start_time:.2f} 秒")

    # --- 结果可视化 ---
    plt.xlabel('初始攻击移除的节点比例 (1-p)')
    plt.ylabel('最终相互连接的巨型分量大小 (P∞)')
    plt.title(f'不同网络类型的级联失效过程 (N={N}, <k>={AVG_DEGREE})')
    plt.grid(True)
    plt.legend()
    plt.show()
