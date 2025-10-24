import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

def get_giant_component(G):
    """返回图G的最大连通分量"""
    if not G.nodes():
        return nx.Graph()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = G.subgraph(components[0])
    return giant

def simulate_cascade(G_A, G_B, initial_attack_fraction):
    """
    模拟相互依赖网络的级联失效过程（全网络参与，但考虑连通性）
    """
    net_A = G_A.copy()
    net_B = G_B.copy()
    N = len(G_A.nodes())
    
    # 初始攻击
    nodes_to_remove_A = np.random.choice(
        list(net_A.nodes()), 
        size=int(N * initial_attack_fraction), 
        replace=False
    )
    net_A.remove_nodes_from(nodes_to_remove_A)
    
    last_A_nodes_count = len(net_A.nodes())
    last_B_nodes_count = len(net_B.nodes())
    
    while True:
        # 1. A网络内部失效：只保留最大连通分量
        if len(net_A.nodes()) > 0:
            giant_A = get_giant_component(net_A)
            nodes_to_remove_in_A = set(net_A.nodes()) - set(giant_A.nodes())
            if nodes_to_remove_in_A:
                net_A.remove_nodes_from(nodes_to_remove_in_A)
        
        # 2. 跨网络依赖失效 (A -> B)
        failed_nodes_in_A = set(G_A.nodes()) - set(net_A.nodes())
        nodes_to_remove_B = set(net_B.nodes()) & failed_nodes_in_A
        if nodes_to_remove_B:
            net_B.remove_nodes_from(nodes_to_remove_B)
        
        # 3. B网络内部失效：只保留最大连通分量
        if len(net_B.nodes()) > 0:
            giant_B = get_giant_component(net_B)
            nodes_to_remove_in_B = set(net_B.nodes()) - set(giant_B.nodes())
            if nodes_to_remove_in_B:
                net_B.remove_nodes_from(nodes_to_remove_in_B)
        
        # 4. 跨网络依赖失效 (B -> A)
        failed_nodes_in_B = set(G_B.nodes()) - set(net_B.nodes())
        nodes_to_remove_A = set(net_A.nodes()) & failed_nodes_in_B
        if nodes_to_remove_A:
            net_A.remove_nodes_from(nodes_to_remove_A)
        
        # 检查是否稳定
        current_A_nodes_count = len(net_A.nodes())
        current_B_nodes_count = len(net_B.nodes())
        if (current_A_nodes_count == last_A_nodes_count and
            current_B_nodes_count == last_B_nodes_count):
            break
        last_A_nodes_count = current_A_nodes_count
        last_B_nodes_count = current_B_nodes_count
    
    final_fraction = len(net_A.nodes()) / N
    return final_fraction

def create_sf_graph(n, gamma, k_avg):
    while True:
        degrees = nx.utils.powerlaw_sequence(n, gamma)
        degrees = [int(d) for d in degrees]
        if sum(degrees) % 2 != 0:
            idx = np.random.randint(0, n)
            degrees[idx] += 1
        current_k_avg = sum(degrees) / n
        if k_avg - 0.5 < current_k_avg < k_avg + 0.5:
            G = nx.configuration_model(degrees)
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            # 保证生成的网络节点数为n
            if G.number_of_nodes() == n:
                return G

if __name__ == '__main__':
    N = 50000
    AVG_DEGREE = 4
    attack_fractions = np.linspace(0, 1, 51)
    num_runs = 3
    network_configs = {
        "ER": {"type": "ER"},
        "RR": {"type": "RR"},
        "SF (γ=3.0)": {"type": "SF", "gamma": 3.0},
        "SF (γ=2.7)": {"type": "SF", "gamma": 2.7},
        "SF (γ=2.3)": {"type": "SF", "gamma": 2.3},
    }
    plt.figure(figsize=(12, 8))
    for name, config in network_configs.items():
        start_time = time.time()
        print(f"\n开始模拟 {name} 网络...")
        print("  正在生成网络 A 和 B...")
        if config["type"] == "ER":
            P = AVG_DEGREE / (N - 1)
            G_A = nx.erdos_renyi_graph(N, P)
            G_B = nx.erdos_renyi_graph(N, P)
        elif config["type"] == "RR":
            d = int(AVG_DEGREE)
            G_A = nx.random_regular_graph(d, N)
            G_B = nx.random_regular_graph(d, N)
        elif config["type"] == "SF":
            gamma = config["gamma"]
            if gamma == 3.0:
                m = int(AVG_DEGREE / 2)
                G_A = nx.barabasi_albert_graph(N, m)
                G_B = nx.barabasi_albert_graph(N, m)
            else:
                G_A = create_sf_graph(N, gamma, AVG_DEGREE)
                G_B = create_sf_graph(N, gamma, AVG_DEGREE)


        # 不再只取最大连通分量，整个网络参与
        # 统一节点标签为0~N-1，保证A、B网络节点一一对应且不丢失节点
        # G_A = nx.convert_node_labels_to_integers(G_A, first_label=0)
        # G_B = nx.convert_node_labels_to_integers(G_B, first_label=0)
            
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
        plt.plot(attack_fractions, final_sizes, 'o-', label=name)
        end_time = time.time()
        print(f"  {name} 模拟完成，耗时: {end_time - start_time:.2f} 秒")


    plt.xlabel('初始攻击移除的节点比例 (1-p)')
    plt.ylabel('最终相互连接的节点比例 (P∞)')
    plt.title(f'不同网络类型的级联失效过程（全网络参与）(N={N}, <k>={AVG_DEGREE})')
    plt.grid(True)
    plt.legend()
    plt.show()
