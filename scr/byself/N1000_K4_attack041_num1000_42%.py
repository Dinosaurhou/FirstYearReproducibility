import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import json

rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 会返回级联失效之后网络A中最大连通分量的节点比例
def cascade_failure(G1, G2, dependency_map, initial_removal_fraction):
    '''
    模拟相互依赖网络的级联失效过程
    参数:
        G_A: 网络A (会被修改) 
        G_B: 网络B (会被修改)
        dependency_map: 依赖关系映射
        initial_removal_fraction: 初始移除的节点比例
    '''
    
    # 复制网络以避免修改原始网络
    G1 = G1.copy()
    G2 = G2.copy()
    n = G1.number_of_nodes()

    # TODO: 要注意如果攻击节点比例直接是1的话,整个网络都会被移除,就直接可以返回了
    num_to_remove = int(n * initial_removal_fraction)
    if num_to_remove >= n:
        return 0.0
    
    # 步骤1: 初始随机攻击网络A
    nodes_to_remove = np.random.choice(G1.nodes(), size=num_to_remove, replace=False)
    # 移除网络A中的节点
    G1.remove_nodes_from(nodes_to_remove)

    # 步骤2: 更新网络B，初始移除网络B的节点
    for node in nodes_to_remove:
        if node in dependency_map:
            dependent_nodes = dependency_map[node]
            G2.remove_node(dependent_nodes)


    # 去记录两个网络有没有边被移除，F是这次没有移除边，T是有边被移除
    flag1 = True
    flag2 = True
    # 级联失效过程
    # 都没有边被移除就停止，也就是F && F 退出
    while flag1 or flag2:

        # 应该在for循环之后判断这次有没有边被移除，否则flag2会被覆盖掉
        flag2 = False
        # 遍历网络B中的每条边找出边连接的两个节点，
        # 通过这两个节点找出对应的网络A中的两个节点，
        # 判断这两个节点是否在同一个连通分量中
        edges_to_list2 = list(G2.edges())
        for edge in edges_to_list2:
            node_b1, node_b2 = edge
            # 通过dependency_map找到对应网络A中的节点
            # dependency_map: key为网络A的节点，value为网络B的节点
            # 需要反向查找：从网络B的节点找到网络A的节点
            node_a1 = None
            node_a2 = None
            for a_node, b_node in dependency_map.items():
                if b_node == node_b1:
                    node_a1 = a_node
                if b_node == node_b2:
                    node_a2 = a_node
            # 判断这两个节点是否都存在于网络A中，且是否在同一个连通分量中
            if node_a1 in G1.nodes() and node_a2 in G1.nodes():
                # 使用nx.node_connected_component获取节点所在的连通分量
                component_a1 = nx.node_connected_component(G1, node_a1)
                if node_a2 not in component_a1:
                    # 不在同一个连通分量中，移除网络B中的这条边
                    G2.remove_edge(node_b1, node_b2)
                    flag2 = True
        
        flag1 = False
        # 同样的操作，遍历网络A中的每条边
        edges_to_list1 = list(G1.edges())
        for edge in edges_to_list1:
            node_a1, node_a2 = edge
            # 通过dependency_map找到对应网络B中的节点
            node_b1 = None
            node_b2 = None
            node_b1 = dependency_map.get(node_a1)
            node_b2 = dependency_map.get(node_a2)
            # 判断这两个节点是否都存在于网络B中，且是否在同一个连通分量中
            if node_b1 in G2.nodes() and node_b2 in G2.nodes():
                component_b1 = nx.node_connected_component(G2, node_b1)
                if node_b2 not in component_b1:
                    # 不在同一个连通分量中，移除网络A中的这条边
                    G1.remove_edge(node_a1, node_a2)
                    flag1 = True
    return G1, G2


if __name__ == "__main__":
    
    # 平均度 = 4，对于ER网络，p = <k>/(N-1)
    N = 1000
    average_degree = 4
    p = average_degree / (N - 1)

    # 创建ER随机图 - 网络A
    G_A = nx.erdos_renyi_graph(N, p)
    # 创建ER随机图 - 网络B（使用不同的随机种子）
    G_B = nx.erdos_renyi_graph(N, p)

    # 打印网络A基本信息
    print("=== 网络A ===")
    print(f"节点数: {G_A.number_of_nodes()}")
    print(f"边数: {G_A.number_of_edges()}")
    print(f"实际平均度: {2 * G_A.number_of_edges() / G_A.number_of_nodes():.2f}")

    # 打印网络B基本信息
    print("\n=== 网络B ===")
    print(f"节点数: {G_B.number_of_nodes()}")
    print(f"边数: {G_B.number_of_edges()}")
    print(f"实际平均度: {2 * G_B.number_of_edges() / G_B.number_of_nodes():.2f}")

    # 存储节点对应关系的字典：key为网络A的节点，value为网络B的对应节点
    node_mapping = {i: i for i in range(N)}
    print(f"\n节点对应关系: {list(node_mapping.items())[:5]}...")  # 显示前5个


    # 固定攻击比例为0.41
    initial_removal_fraction = 0.41
    num_experiments = 1000  # 模拟1000次

    print(f"\n{'='*50}")
    print(f"初始攻击比例: {initial_removal_fraction}")
    print(f"模拟次数: {num_experiments}")
    print(f"{'='*50}\n")

    exitence_count = 0  # 记录巨片存在的次数
    
    # 进行1000次实验
    for exp_num in range(num_experiments):
        
        GA_after, GB_after = cascade_failure(G_A, G_B, node_mapping, initial_removal_fraction)

        # 获取网络A中的最大连通分量
        if GA_after.number_of_nodes() > 0:
            largest_cc_A = max(nx.connected_components(GA_after), key=len)
            largest_cc_size = len(largest_cc_A)
        else:
            largest_cc_size = 0

        # 如果这个连通分量的大小>=10，就记录说明这次在此攻击比例下巨片存在
        if largest_cc_size >= 10:
            exitence_count += 1
            if (exp_num + 1) % 100 == 0:
                print(f"实验 {exp_num + 1}/{num_experiments}: 巨片存在，大小为 {largest_cc_size}")
        else:
            if (exp_num + 1) % 100 == 0:
                print(f"实验 {exp_num + 1}/{num_experiments}: 巨片不存在")

    # 计算巨片存在的概率
    p_giant = exitence_count / num_experiments

    # 输出最终结果
    print(f"\n{'='*50}")
    print("实验结果:")
    print(f"{'='*50}")
    print(f"攻击比例: {initial_removal_fraction}")
    print(f"总实验次数: {num_experiments}")
    print(f"巨片存在次数: {exitence_count}")
    print(f"巨片存在概率: {p_giant:.4f} ({p_giant*100:.2f}%)")
    print(f"{'='*50}")

    # 保存数据到文件
    save_dir = "scr/byself"
    os.makedirs(save_dir, exist_ok=True)

    # 保存数据为JSON格式
    data = {
        'initial_removal_fraction': initial_removal_fraction,
        'p_giant': p_giant,
        'exitence_count': exitence_count,
        'parameters': {
            'N': N,
            'average_degree': average_degree,
            'num_experiments': num_experiments
        }
    }

    file_path = os.path.join(save_dir, 'cascade_failure_attack041_num1000.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n数据已保存到: {file_path}")


