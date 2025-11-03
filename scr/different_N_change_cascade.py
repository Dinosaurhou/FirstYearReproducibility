import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def foreign_neighbors(node, G):
    """查找所有最初来自不同网络的邻居节点
    
    参数:
    - node: 图中的一个节点
    - G: networkx图
    
    返回:
    - set: 外部邻居节点集合
    """
    if node not in G.nodes():
        return set()
    
    foreign = []
    numb = G.nodes[node]['layer']
    s = set(G.neighbors(node))
    while s:
        x = s.pop()
        if x in G.nodes() and G.nodes[x]['layer'] != numb:
            foreign.append(x)
    
    return set(foreign)

def cascade_fail(G, g1, g2, target, verbose=False):
    """移除目标节点及其来自另一个网络的邻居节点
    
    参数:
    - G, g1, g2: networkx图
    - target: 目标节点
    - verbose: 是否显示详细信息
    
    返回:
    - G, g1, g2: 更新后的图
    """
    if target not in G.nodes():
        return G, g1, g2
    
    # 如果是相互依赖网络
    interconnected = (len(g1.nodes()) != 0 and len(g2.nodes()) != 0)
    if interconnected:
        num = G.nodes[target]['layer']
        foreign_nodes = foreign_neighbors(target, G)
        if foreign_nodes:
            for neigh in foreign_nodes:
                if neigh in G.nodes():
                    G.remove_node(neigh)
                if num == 2 and neigh in g1.nodes():
                    g1.remove_node(neigh)
                elif num == 1 and neigh in g2.nodes():
                    g2.remove_node(neigh)
                if verbose:
                    print('Deleted neighbour', neigh)
    
    # 移除目标节点
    if target in G.nodes():
        G.remove_node(target)
    if interconnected:
        if num == 1 and target in g1.nodes():
            g1.remove_node(target)
        elif num == 2 and target in g2.nodes():
            g2.remove_node(target)
    
    return G, g1, g2

def cascade_rec(G, g1, g2, counter, verbose=False):
    """递归移除远距离节点的边
    
    参数:
    - G, g1, g2: networkx图
    - counter: 计数器
    - verbose: 是否显示详细信息
    
    返回:
    - G, g1, g2: 更新后的图
    """
    removed = 0
    
    # 获取g2中的边
    edges = list(g2.edges())
    
    # 获取g1的连通分量列表
    components = list(nx.connected_components(g1))
    
    # 对于每条边(a,b)，找到节点a和b的外部邻居
    # 如果这些邻居在不同的簇中，删除边(a,b)
    for a, b in edges:
        if not G.has_edge(a, b):
            continue
            
        n1 = foreign_neighbors(a, G)
        n2 = foreign_neighbors(b, G)
        if not n1 or not n2:
            continue
            
        for comp in components:
            if (n1.issubset(comp) and not n2.issubset(comp)) or (not n1.issubset(comp) and n2.issubset(comp)):
                if G.has_edge(a, b):
                    G.remove_edge(a, b)
                if g2.has_edge(a, b):
                    g2.remove_edge(a, b)
                
                removed = 1
                if verbose:
                    print('Removed', tuple((a, b)))
                break
    
    # 如果移除了边，切换视角继续查找
    if removed == 1:
        G, g1, g2 = cascade_rec(G, g2, g1, 1, verbose)
    
    # 如果不成功，再次切换视角，但减少计数器以最终停止递归
    if removed == 0 and counter > 0:
        G, g1, g2 = cascade_rec(G, g2, g1, counter - 1, verbose)
    
    return G, g1, g2

def simulate_cascade(G_A, G_B, initial_attack_fraction):
    """
    模拟相互依赖网络的级联失效过程
    
    参数:
    - G_A, G_B: 两个networkx图对象，代表相互依赖的网络
    - initial_attack_fraction: 在网络A中初始移除的节点比例 (0到1之间)
    
    返回:
    - 最终稳定后，网络中剩余节点的比例
    """
    N = len(G_A.nodes())
    
    # 创建联合图G，为了避免节点ID冲突，给网络B的节点加上偏移量
    G = nx.Graph()
    
    # 添加网络A的节点和边，标记为layer 1
    for node in G_A.nodes():
        G.add_node(node, layer=1)
    G.add_edges_from(G_A.edges())
    
    # 添加网络B的节点和边，标记为layer 2
    # 给网络B的节点加上偏移量N，避免与网络A的节点ID冲突
    for node in G_B.nodes():
        G.add_node(node + N, layer=2)
    for u, v in G_B.edges():
        G.add_edge(u + N, v + N)
    
    # 添加跨网络的依赖边（节点i in A 依赖于 节点i in B）
    for node in range(N):
        if node in G_A.nodes() and node in G_B.nodes():
            G.add_edge(node, node + N)
    
    # 创建子图副本
    g1 = nx.Graph()
    g2 = nx.Graph()
    
    # 为g1添加网络A的节点和边
    for node in G_A.nodes():
        g1.add_node(node, layer=1)
    g1.add_edges_from(G_A.edges())
    
    # 为g2添加网络B的节点和边（使用偏移后的ID）
    for node in G_B.nodes():
        g2.add_node(node + N, layer=2)
    for u, v in G_B.edges():
        g2.add_edge(u + N, v + N)
    
    # --- 1. 初始攻击 ---
    # 随机选择要攻击的节点
    candidates = set()
    p = 1 - initial_attack_fraction  # 保留比例
    
    for node in list(g1.nodes()):
        if np.random.random() < 1 - p:
            candidates.add(node)
    
    # --- 2. 级联失效 ---
    # 删除节点并更新集合
    while candidates:
        target = candidates.pop()
        G, g1, g2 = cascade_fail(G, g1, g2, target=target, verbose=False)
        nodes_updated = set(G.nodes())
        candidates.intersection_update(nodes_updated)
    
    # --- 3. 递归检测簇并移除边 ---
    if len(g1.nodes()) > 0 and len(g2.nodes()) > 0:
        G, g1, g2 = cascade_rec(G, g1, g2, 1, verbose=False)
    
    # --- 4. 计算最终剩余节点比例 ---
    # 统计layer 1中剩余的节点数
    remaining_nodes = sum(1 for node in G.nodes() if G.nodes[node]['layer'] == 1)
    final_fraction = remaining_nodes / N
    
    return final_fraction


if __name__ == '__main__':
    # --- 网络参数 ---
    N_values = [1000, 2000, 4000, 8000]
    AVG_DEGREE = 4  # 平均度保持不变

    # --- 模拟参数 ---
    attack_fractions = np.linspace(0, 1, 51)
    retained_fractions = 1 - attack_fractions
    num_runs = 5

    # --- 新的横坐标 ---
    x_axis_values = retained_fractions * AVG_DEGREE

    # --- 准备绘图 ---
    plt.figure(figsize=(10, 6))
    
    critical_ps = []

    # --- 对每个N值运行模拟 ---
    for N in N_values:
        print(f"\n开始模拟 N = {N} 的网络...")
        P = AVG_DEGREE / (N - 1)

        # --- 生成两个ER网络 ---
        G_A = nx.erdos_renyi_graph(N, P, seed=None)
        G_B = nx.erdos_renyi_graph(N, P, seed=None)

        final_sizes = []
        for p_attack in attack_fractions:
            avg_size = 0
            for i in range(num_runs):
                final_size = simulate_cascade(G_A, G_B, p_attack)
                avg_size += final_size
            avg_size /= num_runs
            final_sizes.append(avg_size)
            print(f"  N={N}, 初始攻击比例 p = {p_attack:.2f}, 最终网络规模比例 = {avg_size:.3f}")

        # --- 寻找当前N的临界点 ---
        diffs = np.diff(final_sizes)
        critical_index = np.argmin(diffs)
        p_c_retained = retained_fractions[critical_index]
        critical_ps.append(p_c_retained)
        print(f"  N={N} 的临界点 p_c ≈ {p_c_retained:.3f}")

        # --- 在图表中绘制当前N值的结果 ---
        plt.plot(x_axis_values, final_sizes, 'o-', label=f'N = {N}')

    # --- 计算并标记平均临界点 ---
    avg_pc = np.mean(critical_ps)
    print(f"\n所有网络规模的平均临界点 p_c ≈ {avg_pc:.3f}")
    avg_pc_scaled = avg_pc * AVG_DEGREE

    # 在图上用垂直虚线标记平均临界点
    plt.axvline(x=avg_pc_scaled, color='r', linestyle='--', label=f'平均临界点 $p_c \\times <k> \\approx {avg_pc_scaled:.3f}$')
    
    # --- 结果可视化 ---
    plt.xlabel('初始保留节点比例与平均度的乘积 (p * <k>)')
    plt.ylabel('最终相互连接的巨型分量大小 (P∞)')
    plt.title(f'不同规模相互依赖网络的级联失效 (平均度 <k> = {AVG_DEGREE})')
    plt.grid(True)
    plt.legend()
    plt.show()