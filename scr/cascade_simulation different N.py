import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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
        net_A.nodes(), 
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
        nodes_to_remove_B = set(net_B.nodes()) & failed_nodes_in_A # 假设节点 i in A 依赖 i in B
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


if __name__ == '__main__':
    # --- 网络参数 ---
    N_values = [1000, 2000, 4000, 8000, 16000, 32000, 64000] # 不同的节点数
    AVG_DEGREE = 4  # 平均度保持不变

    # --- 模拟参数 ---
    attack_fractions = np.linspace(0, 1, 51) # 从 0% 到 100% 的攻击强度, 增加采样点使曲线更平滑
    num_runs = 5 # 为了结果的稳定性，对每个攻击强度进行多次模拟并取平均值

    # --- 准备绘图 ---
    plt.figure(figsize=(10, 6))
    
    critical_ps = [] # 用于存储每个N值对应的临界p值

    # --- 对每个N值运行模拟 ---
    for N in N_values:
        print(f"\n开始模拟 N = {N} 的网络...")
        P = AVG_DEGREE / (N - 1)  # ER网络连接概率随N变化

        # --- 生成两个ER网络 ---
        # 使用不同的种子以创建不同的网络拓扑
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
        # 临界点是网络规模下降最快的地方
        # 我们可以通过计算相邻点之间的差值来找到它
        diffs = np.diff(final_sizes)
        critical_index = np.argmin(diffs)
        p_c = attack_fractions[critical_index]
        critical_ps.append(p_c)
        print(f"  N={N} 的临界点 p_c ≈ {p_c:.3f}")

        # --- 在图表中绘制当前N值的结果 ---
        plt.plot(attack_fractions, final_sizes, 'o-', label=f'N = {N}')

    # --- 计算并标记平均临界点 ---
    avg_pc = np.mean(critical_ps)
    print(f"\n所有网络规模的平均临界点 p_c ≈ {avg_pc:.3f}")

    # 在图上用垂直虚线标记平均临界点
    plt.axvline(x=avg_pc, color='r', linestyle='--', label=f'平均临界点 $p_c \\approx {avg_pc:.3f}$')
    
    # --- 结果可视化 ---
    plt.xlabel('初始攻击移除的节点比例 (1-p)')
    plt.ylabel('最终相互连接的巨型分量大小 (P∞)')
    plt.title(f'不同规模相互依赖网络的级联失效 (平均度 <k> = {AVG_DEGREE})')
    plt.grid(True)
    plt.legend()
    plt.show()