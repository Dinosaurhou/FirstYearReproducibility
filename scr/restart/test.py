import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号

def create_er_network(n, avg_degree):
    """
    创建ER随机网络（不强制连通）
    """
    p = avg_degree / (n - 1)
    G = nx.erdos_renyi_graph(n, p)
    return G

def create_interdependent_networks(n, avg_degree):
    """
    创建两个相互依赖的网络A和B，节点一一对应
    
    参数:
        n: 每个网络的节点数
        avg_degree: 平均度
    
    返回:
        G_A, G_B: 两个网络
        dependency_map: 依赖关系映射 {A节点: B节点}
    """
    G_A = create_er_network(n, avg_degree)
    G_B = create_er_network(n, avg_degree)
    
    # 创建一一对应的依赖关系
    dependency_map = {i: i for i in range(n)}
    
    return G_A, G_B, dependency_map


def get_largest_component_nodes(G):
    """
    获取网络中最大连通分量的节点集合
    
    参数:
        G: networkx图对象
    
    返回:
        最大连通分量的节点集合
    """
    if G.number_of_nodes() == 0:
        return set()
    
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    if not components:
        return set()
    
    # 返回最大的连通分量
    largest_component = max(components, key=len)
    return largest_component


def cascade_failure(G_A, G_B, dependency_map, initial_removal_fraction):
    """
    模拟相互依赖网络的级联失效过程
    
    参数:
        G_A: 网络A (会被修改) 
        G_B: 网络B (会被修改)
        dependency_map: 依赖关系映射
        initial_removal_fraction: 初始移除的节点比例
    
    返回:
        final_size_A: 网络A最终最大连通分量的大小
        final_size_B: 网络B最终最大连通分量的大小
    """
    # 复制网络以避免修改原始网络
    G_A = G_A.copy()
    G_B = G_B.copy()
    n = G_A.number_of_nodes()
    
    # 步骤1: 初始随机攻击网络A
    num_to_remove = int(n * initial_removal_fraction)
    nodes_to_remove = np.random.choice(list(G_A.nodes()), num_to_remove, replace=False)
    
    # 移除初始节点
    active_nodes_A = set(G_A.nodes()) - set(nodes_to_remove)
    active_nodes_B = set(G_B.nodes())
    
    # 开始迭代级联失效过程
    iteration = 0
    # max_iterations = 10000
    max_iterations = max(1000, int(10 * np.log(n)))  # 至少1000次
    
    while iteration < max_iterations:
        iteration += 1
        previous_active_A = active_nodes_A.copy()
        previous_active_B = active_nodes_B.copy()
        
        # 步骤2: 由于依赖关系，网络B中对应的节点也失效
        # 如果A中的节点i失效，则B中对应的节点也失效
        nodes_to_remove_B = set()
        for node_a in list(G_A.nodes()):
            if node_a not in active_nodes_A:
                node_b = dependency_map.get(node_a)
                if node_b is not None and node_b in active_nodes_B:
                    nodes_to_remove_B.add(node_b)
        
        active_nodes_B -= nodes_to_remove_B
        
        # 步骤3: 移除B中失去连通性的节点（不在最大连通分量中的节点）
        if len(active_nodes_B) > 0:
            G_B_temp = G_B.subgraph(active_nodes_B).copy()
            largest_component_B = get_largest_component_nodes(G_B_temp)
            active_nodes_B = largest_component_B
        else:
            active_nodes_B = set()
        
        # 步骤4: 由于依赖关系，A中对应的节点也失效
        # 如果B中的节点失效，则A中对应的节点也失效
        nodes_to_remove_A = set()
        for node_b in list(G_B.nodes()):
            if node_b not in active_nodes_B:
                # 找到依赖于这个B节点的A节点
                for node_a, dep_node_b in dependency_map.items():
                    if dep_node_b == node_b and node_a in active_nodes_A:
                        nodes_to_remove_A.add(node_a)
        
        active_nodes_A -= nodes_to_remove_A
        
        # 步骤5: 移除A中失去连通性的节点（不在最大连通分量中的节点）
        if len(active_nodes_A) > 0:
            G_A_temp = G_A.subgraph(active_nodes_A).copy()
            largest_component_A = get_largest_component_nodes(G_A_temp)
            active_nodes_A = largest_component_A
        else:
            active_nodes_A = set()
        
        # 检查是否收敛（没有新的节点失效）
        if active_nodes_A == previous_active_A and active_nodes_B == previous_active_B:
            break
    
    # 返回最终的最大连通分量大小
    final_size_A = len(active_nodes_A)
    final_size_B = len(active_nodes_B)
    
    return final_size_A, final_size_B


def simulate_cascade_for_different_fractions(n, avg_degree, num_simulations=50):
    """
    对不同的攻击比例进行级联失效模拟
    
    参数:
        n: 网络节点数
        avg_degree: 平均度
        num_simulations: 每个攻击比例的模拟次数
    
    返回:
        fractions: 保留节点比例列表
        avg_sizes_A: 网络A的平均最大连通分量比例
        avg_sizes_B: 网络B的平均最大连通分量比例
    """
    # 攻击比例从0到1
    attack_fractions = np.linspace(0, 1, 21)  # 0, 0.05, 0.1, ..., 1.0
    remaining_fractions = 1 - attack_fractions  # 保留节点比例
    
    avg_sizes_A = []
    avg_sizes_B = []
    std_sizes_A = []
    std_sizes_B = []
    
    for attack_frac in attack_fractions:
        print(f"  模拟攻击比例: {attack_frac:.2f} (保留比例: {1-attack_frac:.2f})")
        
        sizes_A = []
        sizes_B = []
        
        for sim in range(num_simulations):
            # 创建新的网络对
            G_A, G_B, dependency_map = create_interdependent_networks(n, avg_degree)
            
            # 执行级联失效
            final_size_A, final_size_B = cascade_failure(G_A, G_B, dependency_map, attack_frac)
            
            # 记录最大连通分量占原网络的比例
            sizes_A.append(final_size_A / n)
            sizes_B.append(final_size_B / n)
        
        # 计算平均值和标准差 (A和B结果相同,只保留A的结果)
        avg_sizes_A.append(np.mean(sizes_A))
        avg_sizes_B.append(np.mean(sizes_B))
        std_sizes_A.append(np.std(sizes_A))
        std_sizes_B.append(np.std(sizes_B))
    
    # 由于网络A和B完全对称,返回A的结果即可
    return remaining_fractions, avg_sizes_A, std_sizes_A


def plot_results(remaining_fractions, avg_sizes, std_sizes, n):
    """
    绘制单个网络规模的结果图
    
    参数:
        remaining_fractions: 保留节点比例
        avg_sizes: 平均最大连通分量比例
        std_sizes: 标准差
        n: 网络节点数
    """
    plt.figure(figsize=(10, 6))
    
    # 由于网络A和B完全对称,只绘制一条曲线
    plt.plot(remaining_fractions, avg_sizes, 'b-o', label=f'N={n}', linewidth=2, markersize=6)
    plt.fill_between(remaining_fractions, 
                     np.array(avg_sizes) - np.array(std_sizes), 
                     np.array(avg_sizes) + np.array(std_sizes), 
                     alpha=0.2, color='blue')
    
    plt.xlabel('保留节点比例 (1-p)', fontsize=12)
    plt.ylabel('最大连通分量占比 (P∞)', fontsize=12)
    plt.title(f'相互依赖网络的级联失效过程\n(ER网络, N={n}, <k>=4)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    # plt.savefig(f'cascade_failure_N{n}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n图表已保存为 'cascade_failure_N{n}.png'")


def plot_multiple_results(results_dict, avg_degree=4):
    """
    绘制多个网络规模的对比图
    
    参数:
        results_dict: 字典,格式为 {N: (remaining_fractions, avg_sizes, std_sizes)}
        avg_degree: 平均度
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g', 'm']
    markers = ['o', 's', '^', 'D']
    
    for idx, (n, (remaining_fractions, avg_sizes, std_sizes)) in enumerate(sorted(results_dict.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # 计算横坐标: (1-p) × <k>
        x_values = np.array(remaining_fractions) * avg_degree
        
        # 只绘制保留比例在0.5到0.7之间的数据点
        mask = (np.array(remaining_fractions) >= 0.5) & (np.array(remaining_fractions) <= 0.7)
        
        plt.plot(x_values[mask], np.array(avg_sizes)[mask], 
                color=color, marker=marker, label=f'N={n}', 
                linewidth=2, markersize=6)
        plt.fill_between(x_values[mask], 
                         np.array(avg_sizes)[mask] - np.array(std_sizes)[mask], 
                         np.array(avg_sizes)[mask] + np.array(std_sizes)[mask], 
                         alpha=0.15, color=color)
    
    plt.xlabel('(1-p) × <k>', fontsize=12)
    plt.ylabel('最大连通分量占比 (P∞)', fontsize=12)
    plt.title('相互依赖网络的级联失效过程 - 不同网络规模对比\n(ER网络, <k>=4)', fontsize=14)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    # 设置横坐标范围: 0.5×4 到 0.7×4
    plt.xlim([0.5 * avg_degree, 0.7 * avg_degree])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('cascade_failure_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n对比图已保存为 'cascade_failure_comparison.png'")

if __name__ == '__main__':
    
    # 设置参数
    # network_sizes = [250, 500, 1000, 2000]  # 不同的网络规模
    network_sizes = [500, 1000, 2000]
    avg_degree = 4  # 平均度
    num_simulations = 20  # 每个攻击比例的模拟次数
    
    print("=" * 60)
    print("相互依赖网络级联失效模拟 - 多规模对比")
    print("=" * 60)
    print(f"网络规模: {network_sizes}")
    print(f"平均度: {avg_degree}")
    print(f"每个数据点模拟次数: {num_simulations}")
    print("=" * 60)
    print()
    
    # 存储所有结果
    all_results = {}
    
    # 对每个网络规模进行模拟
    for n in network_sizes:
        print(f"\n{'='*60}")
        print(f"开始模拟 N = {n} 的网络")
        print(f"{'='*60}")
        
        # 运行模拟
        remaining_fractions, avg_sizes, std_sizes = \
            simulate_cascade_for_different_fractions(n, avg_degree, num_simulations)
        
        # 保存结果
        all_results[n] = (remaining_fractions, avg_sizes, std_sizes)
        
        # 绘制单个结果（如果需要）
        # plot_results(remaining_fractions, avg_sizes, std_sizes, n, avg_degree)
        
        # 打印关键结果
        print("\n" + "-" * 60)
        print(f"N={n} 的关键结果:")
        print("-" * 60)
        print(f"{'保留比例':<15} {'巨连通分量占比':<15}")
        print("-" * 60)
        for i in range(0, len(remaining_fractions), 2):
            print(f"{remaining_fractions[i]:<15.2f} {avg_sizes[i]:<15.3f}")
        print("-" * 60)
    
    # 绘制所有结果的对比图
    print("\n" + "=" * 60)
    print("生成对比图...")
    print("=" * 60)
    plot_multiple_results(all_results, avg_degree)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("模拟完成!")
    print("=" * 60)
    print(f"已生成 {len(network_sizes)} 个单独的结果图和 1 个对比图")
    print("=" * 60)
