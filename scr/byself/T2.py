import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

rcParams['axes.unicode_minus'] = False  # 正常显示负号

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

    # TODO: 要注意如果攻击节点比例直接是1的话，整个网络都会被移除，就直接可以返回了

    # 步骤1: 初始随机攻击网络A
    num_to_remove = int(n * initial_removal_fraction)
    nodes_to_remove = np.random.choice(G1.nodes(), size=num_to_remove, replace=False)
    # 移除网络A中的节点
    G1.remove_nodes_from(nodes_to_remove)

    # 步骤2: 更新网络B，初始移除网络B的节点
    for node in nodes_to_remove:
        if node in dependency_map:
            dependent_nodes = dependency_map[node]
            G2.remove_node(dependent_nodes)




    # XXX # 初步处理：给网络A分簇群
    # # 找出所有连通分量并编号
    # connected_components = list(nx.connected_components(G1))
    # # 创建一个字典：key为分量编号，value为该分量的节点集合
    # component_dict = {i: component for i, component in enumerate(connected_components)}
    # # 创建一个字典：key为节点，value为该节点所属的分量编号
    # node_to_component = {}
    # for component_id, nodes in component_dict.items():
    #     # 告诉我节点我就知道属于哪个连通分量
    #     for node in nodes:
    #         node_to_component[node] = component_id

    
    # 去记录两个网络有没有边被移除
    flag1 = False
    flag2 = False
    # 级联失效过程
    while flag1 or flag2:

        # 遍历网络B中的每条边找出边连接的两个节点，
        # 通过这两个节点找出对应的网络A中的两个节点，
        # 判断这两个节点是否在同一个连通分量中
        for edge in G2.edges():
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
        
        # 同样的操作，遍历网络A中的每条边
        for edge in G1.edges():
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

        











if __name__ == "__main__":
    # 生成ER网络
    # N = 100 节点数
    # 平均度 = 4，对于ER网络，p = <k>/(N-1)
    N = 100
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

    cascade_failure(G_A, G_B, node_mapping, initial_removal_fraction=0.1)

    # 3D可视化
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 为两个网络生成2D布局
    pos_A = nx.spring_layout(G_A, seed=42)
    pos_B = nx.spring_layout(G_B, seed=123)

    # 将2D位置转换为3D坐标
    # 网络A在z=0平面
    pos_A_3d = {node: (pos[0], pos[1], 0) for node, pos in pos_A.items()}
    # 网络B在z=1平面
    pos_B_3d = {node: (pos[0], pos[1], 1) for node, pos in pos_B.items()}

    # 绘制网络A的边
    for edge in G_A.edges():
        x = [pos_A_3d[edge[0]][0], pos_A_3d[edge[1]][0]]
        y = [pos_A_3d[edge[0]][1], pos_A_3d[edge[1]][1]]
        z = [pos_A_3d[edge[0]][2], pos_A_3d[edge[1]][2]]
        ax.plot(x, y, z, c='blue', alpha=0.3, linewidth=0.5)

    # 绘制网络B的边
    for edge in G_B.edges():
        x = [pos_B_3d[edge[0]][0], pos_B_3d[edge[1]][0]]
        y = [pos_B_3d[edge[0]][1], pos_B_3d[edge[1]][1]]
        z = [pos_B_3d[edge[0]][2], pos_B_3d[edge[1]][2]]
        ax.plot(x, y, z, c='red', alpha=0.3, linewidth=0.5)

    # 绘制对应关系的虚线
    for node_a, node_b in node_mapping.items():
        x = [pos_A_3d[node_a][0], pos_B_3d[node_b][0]]
        y = [pos_A_3d[node_a][1], pos_B_3d[node_b][1]]
        z = [pos_A_3d[node_a][2], pos_B_3d[node_b][2]]
        ax.plot(x, y, z, 'k--', alpha=0.1, linewidth=0.5)

    # 绘制网络A的节点
    nodes_A = np.array([pos_A_3d[node] for node in G_A.nodes()])
    ax.scatter(nodes_A[:, 0], nodes_A[:, 1], nodes_A[:, 2], 
            c='lightblue', s=100, alpha=0.8, edgecolors='blue', label='网络A')

    # 绘制网络B的节点
    nodes_B = np.array([pos_B_3d[node] for node in G_B.nodes()])
    ax.scatter(nodes_B[:, 0], nodes_B[:, 1], nodes_B[:, 2], 
            c='lightcoral', s=100, alpha=0.8, edgecolors='red', label='网络B')

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (网络层)')
    ax.set_title(f'双层网络3D可视化\n网络A (蓝色, z=0) 和 网络B (红色, z=1)\n虚线表示节点对应关系', fontsize=14)
    ax.legend()

    # 设置视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()