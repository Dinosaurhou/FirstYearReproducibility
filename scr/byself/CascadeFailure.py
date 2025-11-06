import networkx as nx
import numpy as np
from typing import overload



# 指定受到攻击的节点比例
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



# 指定受到攻击的特定节点
def cascade_failure_specific_nodes(G1, G2, dependency_map, nodes_to_attack : list):
    '''
    模拟相互依赖网络的级联失效过程 - 攻击特定节点版本
    参数:
        G1: 网络A (会被修改) 
        G2: 网络B (会被修改)
        dependency_map: 依赖关系映射
        nodes_to_attack: 要攻击的特定节点列表
    '''
    
    # 复制网络以避免修改原始网络
    G1 = G1.copy()
    G2 = G2.copy()

    # 检查要攻击的节点是否有效
    nodes_to_remove = [node for node in nodes_to_attack if node in G1.nodes()]
    if not nodes_to_remove:
        return G1, G2
    
    # 步骤1: 移除指定的节点
    G1.remove_nodes_from(nodes_to_remove)

    # 步骤2: 更新网络B，移除依赖于被攻击节点的节点
    for node in nodes_to_remove:
        if node in dependency_map:
            dependent_nodes = dependency_map[node]
            if dependent_nodes in G2.nodes():
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

