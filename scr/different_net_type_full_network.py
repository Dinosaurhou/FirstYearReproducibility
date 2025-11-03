import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys

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
    
    # 建立一一对应的依赖关系（关键修正）
    nodes_A = list(G_A.nodes())
    nodes_B = list(G_B.nodes())
    dependency_map = dict(zip(nodes_A, nodes_B))  # A[i] -> B[i]
    reverse_map = dict(zip(nodes_B, nodes_A))     # B[i] -> A[i]
    
    # 初始随机攻击
    nodes_to_remove_A = np.random.choice(
        nodes_A, 
        size=int(N * initial_attack_fraction), 
        replace=False
    )
    net_A.remove_nodes_from(nodes_to_remove_A)
    
    converged = False
    max_iterations = 100
    iteration = 0
    
    while not converged and iteration < max_iterations:
        iteration += 1
        old_A_size = net_A.number_of_nodes()
        old_B_size = net_B.number_of_nodes()
        
        # 步骤1: A网络失效 → B网络依赖失效
        failed_in_A = set(nodes_A) - set(net_A.nodes())
        dependent_fail_B = [dependency_map[n] for n in failed_in_A 
                           if n in dependency_map and dependency_map[n] in net_B.nodes()]
        net_B.remove_nodes_from(dependent_fail_B)
        
        # 步骤2: B网络保留最大连通分量
        if net_B.number_of_nodes() > 0:
            giant_B = max(nx.connected_components(net_B), key=len)
            isolated_B = set(net_B.nodes()) - giant_B
            net_B.remove_nodes_from(isolated_B)
        
        # 步骤3: B网络失效 → A网络依赖失效
        failed_in_B = set(nodes_B) - set(net_B.nodes())
        dependent_fail_A = [reverse_map[n] for n in failed_in_B 
                           if n in reverse_map and reverse_map[n] in net_A.nodes()]
        net_A.remove_nodes_from(dependent_fail_A)
        
        # 步骤4: A网络保留最大连通分量
        if net_A.number_of_nodes() > 0:
            giant_A = max(nx.connected_components(net_A), key=len)
            isolated_A = set(net_A.nodes()) - giant_A
            net_A.remove_nodes_from(isolated_A)
        
        # 检查收敛
        if (net_A.number_of_nodes() == old_A_size and 
            net_B.number_of_nodes() == old_B_size):
            converged = True
    
    return net_A.number_of_nodes() / N

def create_sf_graph(n, gamma, k_avg, max_attempts=100):
    """使用更稳定的幂律网络生成方法"""
    for attempt in range(max_attempts):
        # 生成幂律度序列
        degrees = []
        k_min = 2  # 最小度
        k_max = int(n ** (1/(gamma-1)))  # 理论最大度
        
        for _ in range(n):
            k = np.random.pareto(gamma - 1) * k_min
            k = min(int(k), k_max)
            k = max(k, k_min)
            degrees.append(k)
        
        # 调整使度序列和为偶数
        if sum(degrees) % 2 != 0:
            degrees[np.random.randint(n)] += 1
        
        # 检查平均度
        current_avg = np.mean(degrees)
        if abs(current_avg - k_avg) < 0.3:
            try:
                G = nx.configuration_model(degrees)
                G = nx.Graph(G)  # 移除多重边
                G.remove_edges_from(nx.selfloop_edges(G))
                
                # 验证连通性
                if nx.is_connected(G):
                    return G
            except:
                continue
    
    # 失败后使用BA模型作为后备
    print(f"警告: γ={gamma}网络生成失败，使用BA模型")
    return nx.barabasi_albert_graph(n, int(k_avg/2))

if __name__ == '__main__':
    N = 10000
    AVG_DEGREE = 4
    attack_fractions = np.linspace(0, 1, 51)
    num_runs = 20  # 增加到至少20次
    network_configs = {
        "ER": {"type": "ER"},
        "RR": {"type": "RR"},
        "SF (γ=3.0)": {"type": "SF", "gamma": 3.0},
        "SF (γ=2.7)": {"type": "SF", "gamma": 2.7},
        "SF (γ=2.3)": {"type": "SF", "gamma": 2.3},
    }
    plt.figure(figsize=(12, 8))
    
    # 为所有网络类型添加总进度条
    for name, config in tqdm(network_configs.items(), desc="总进度", position=0):
        start_time = time.time()
        print(f"\n开始模拟 {name} 网络...")
        print("  正在生成网络 A 和 B...")
        sys.stdout.flush()
        
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

        print(f"  网络生成完毕。网络A节点数: {len(G_A.nodes())}, 网络B节点数: {len(G_B.nodes())}")
        sys.stdout.flush()
        
        final_sizes = []
        # 为每个网络类型的攻击比例添加子进度条
        for p_attack in tqdm(attack_fractions, desc=f"{name} 攻击进度", position=1, leave=False):
            avg_size = 0
            for i in range(num_runs):
                final_size = simulate_cascade(G_A, G_B, p_attack)
                avg_size += final_size
            avg_size /= num_runs
            final_sizes.append(avg_size)
        
        plt.plot(attack_fractions, final_sizes, 'o-', label=name)
        end_time = time.time()
        print(f"  {name} 模拟完成，耗时: {end_time - start_time:.2f} 秒")
        sys.stdout.flush()

    plt.xlabel('初始攻击移除的节点比例 (1-p)')
    plt.ylabel('最终相互连接的节点比例 (P∞)')
    plt.title(f'不同网络类型的级联失效过程（全网络参与）(N={N}, <k>={AVG_DEGREE})')
    plt.grid(True)
    plt.legend()
    plt.show()
