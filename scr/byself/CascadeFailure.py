import networkx as nx
import numpy as np


# 指定受到攻击的节点比例（按照簇逻辑删除对应网络的边）
def cascade_failure_fig2(G1, G2, dependency_map, initial_removal_fraction):
    '''
    模拟相互依赖网络的级联失效过程
    参数:
        G1: 网络A (会被修改) 
        G2: 网络B (会被修改)
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

# 指定受到攻击的节点比例（按照簇逻辑删除对应网络的边）
# TODO:未实现添加记录级联阶段stage的功能
def cascade_failure_fig2_stage(G1, G2, dependency_map, initial_removal_fraction):
    '''
    模拟相互依赖网络的级联失效过程
    参数:
        G1: 网络A (会被修改) 
        G2: 网络B (会被修改)
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


# 指定受到攻击的节点比例（选取每个阶段最大的连通分量）
def cascade_failure_max(G1, G2, dependency_map, initial_removal_fraction):
    # 复制网络以避免修改原始网络
    G1 = G1.copy()
    G2 = G2.copy()
    n = G1.number_of_nodes()

    # 记录每个阶段巨片大小变化的列表
    history = []

    # TODO: 要注意如果攻击节点比例直接是1的话，整个网络都会被移除，就直接可以返回了
    num_to_remove = int(n * initial_removal_fraction)
    if num_to_remove >= n:
        G1.clear()
        G2.clear()
        history.append({'stage': 1, 'network': 'A', 'ratio': 0.0})
        history.append({'stage': 2, 'network': 'B', 'ratio': 0.0})
        return G1, G2, history
    
    # 步骤1: 初始随机攻击网络A
    nodes_to_remove = np.random.choice(G1.nodes(), size=num_to_remove, replace=False)
    # 移除网络A中的节点
    G1.remove_nodes_from(nodes_to_remove)

    # 步骤2: 更新网络B，初始移除网络B的节点
    for node in nodes_to_remove:
        if node in dependency_map:
            dependent_nodes = dependency_map[node]
            G2.remove_node(dependent_nodes)

    # 记录每次迭代开始时节点的数量，用于判断是否达到稳定
    last_A_nodes_count = len(G1.nodes())
    last_B_nodes_count = len(G2.nodes())
    current_A_nodes_count = None
    current_B_nodes_count = None

    # 记录阶段数
    stage = 0

    # 级联失效过程
    while True:
        stage += 1
        # 阶段1: 网络A内部失效，保留最大连通分量
        nodes_to_remove_A = set()
        if G1.number_of_nodes() > 0:
            largest_cc_A = max(nx.connected_components(G1), key=len)
            nodes_to_remove_A = set(G1.nodes()) - set(largest_cc_A)
            if nodes_to_remove_A:
                G1.remove_nodes_from(nodes_to_remove_A)
        
        ratio_A = G1.number_of_nodes() / n
        history.append({'stage': stage, 'network': 'A', 'ratio': ratio_A / (1 - initial_removal_fraction)})
        print(f"阶段 {stage} 完成网络A内部失效，移除节点数: {len(nodes_to_remove_A)}， 网络A剩余节点数: {G1.number_of_nodes()}，巨片存在比例为 {ratio_A:.4f}")

        stage += 1
        # 阶段2: 跨网络依赖失效 (A -> B)
        nodes_to_remove_B = set(G2.nodes()) & nodes_to_remove_A # 假设节点 i in A 依赖 i in B
        if nodes_to_remove_B:
            G2.remove_nodes_from(nodes_to_remove_B)
        
        # 网络B内部失效，保留最大连通分量
        nodes_to_remove_B_internal = set()
        if G2.number_of_nodes() > 0:
            largest_cc_B = max(nx.connected_components(G2), key=len)
            nodes_to_remove_B_internal = set(G2.nodes()) - set(largest_cc_B)
            if nodes_to_remove_B_internal:
                G2.remove_nodes_from(nodes_to_remove_B_internal) 
        # 记录网络B巨片比例，已经删除了巨片之外的节点
        ratio_B = G2.number_of_nodes() / n
        history.append({'stage': stage, 'network': 'B', 'ratio': ratio_B / (1 - initial_removal_fraction)})
        print(f"阶段 {stage} 完成网络B内部失效，移除节点数: {len(nodes_to_remove_B_internal)}， 网络B剩余节点数: {G2.number_of_nodes()}，巨片存在比例为 {ratio_B:.4f}")

        # 保留网络B最大连通分量之后，更新网络A
        nodes_to_remove_A_from_B = set(G1.nodes()) & nodes_to_remove_B_internal # 假设节点 i in A 依赖 i in B
        if nodes_to_remove_A_from_B:
            G1.remove_nodes_from(nodes_to_remove_A_from_B)

        # 检查是否达到稳定状态
        current_A_nodes_count = len(G1.nodes())
        current_B_nodes_count = len(G2.nodes())
        if current_A_nodes_count == last_A_nodes_count and current_B_nodes_count == last_B_nodes_count:
            break

        last_A_nodes_count = current_A_nodes_count
        last_B_nodes_count = current_B_nodes_count

    return G1, G2, history

# 修改stage的计数逻辑，我在第一次移除节点数为0的时候stage+1，随即锁住可以退出
def cascade_failure_max_change_stagecount(G1, G2, dependency_map, initial_removal_fraction):
    # 复制网络以避免修改原始网络
    G1 = G1.copy()
    G2 = G2.copy()
    n = G1.number_of_nodes()

    # 记录每个阶段巨片大小变化的列表
    history = []

    # TODO: 要注意如果攻击节点比例直接是1的话，整个网络都会被移除，就直接可以返回了
    num_to_remove = int(n * initial_removal_fraction)
    if num_to_remove >= n:
        G1.clear()
        G2.clear()
        history.append({'stage': 1, 'network': 'A', 'ratio': 0.0})
        history.append({'stage': 2, 'network': 'B', 'ratio': 0.0})
        return G1, G2, history
    
    # 步骤1: 初始随机攻击网络A
    nodes_to_remove = np.random.choice(G1.nodes(), size=num_to_remove, replace=False)
    # 移除网络A中的节点
    G1.remove_nodes_from(nodes_to_remove)

    # 步骤2: 更新网络B，初始移除网络B的节点
    for node in nodes_to_remove:
        if node in dependency_map:
            dependent_nodes = dependency_map[node]
            G2.remove_node(dependent_nodes)

    # 记录每次迭代开始时节点的数量，用于判断是否达到稳定
    last_A_nodes_count = len(G1.nodes())
    last_B_nodes_count = len(G2.nodes())
    current_A_nodes_count = None
    current_B_nodes_count = None

    # 记录阶段数
    stage = 0
    # flag用来锁住stage的增加
    stage_lock = True

    # 级联失效过程
    while stage_lock:
        if stage_lock:
            stage += 1
        # 阶段1: 网络A内部失效，保留最大连通分量
        nodes_to_remove_A = set()
        if G1.number_of_nodes() > 0:
            largest_cc_A = max(nx.connected_components(G1), key=len)
            nodes_to_remove_A = set(G1.nodes()) - set(largest_cc_A)
            if nodes_to_remove_A:
                G1.remove_nodes_from(nodes_to_remove_A)
        
        ratio_A = G1.number_of_nodes() / n
        history.append({'stage': stage, 'network': 'A', 'ratio': ratio_A / (1 - initial_removal_fraction)})
        print(f"阶段 {stage} 完成网络A内部失效，移除节点数: {len(nodes_to_remove_A)}， 网络A剩余节点数: {G1.number_of_nodes()}，巨片存在比例为 {ratio_A:.4f}")

        
        # 阶段2: 跨网络依赖失效 (A -> B)
        # 论文中的依赖关系是节点i in A 依赖 i in B，所以A失效会导致B失效
        # 但是我的dependency_map是A->B，所以A失效会导致B失效
        nodes_to_remove_B_due_to_A = set()
        for node_a in nodes_to_remove_A:
            if node_a in dependency_map:
                nodes_to_remove_B_due_to_A.add(dependency_map[node_a])
        
        if nodes_to_remove_B_due_to_A:
            G2.remove_nodes_from(nodes_to_remove_B_due_to_A)
        
        if stage_lock:
            stage += 1
        # 网络B内部失效，保留最大连通分量
        nodes_to_remove_B_internal = set()
        if G2.number_of_nodes() > 0:
            largest_cc_B = max(nx.connected_components(G2), key=len)
            nodes_to_remove_B_internal = set(G2.nodes()) - set(largest_cc_B)
            if nodes_to_remove_B_internal:
                G2.remove_nodes_from(nodes_to_remove_B_internal) 
        # 记录网络B巨片比例，已经删除了巨片之外的节点
        ratio_B = G2.number_of_nodes() / n
        history.append({'stage': stage, 'network': 'B', 'ratio': ratio_B / (1 - initial_removal_fraction)})
        print(f"阶段 {stage} 完成网络B内部失效，移除节点数: {len(nodes_to_remove_B_internal)}， 网络B剩余节点数: {G2.number_of_nodes()}，巨片存在比例为 {ratio_B:.4f}")

        # 保留网络B最大连通分量之后，更新网络A
        # 反向查找 B -> A 的依赖关系
        nodes_to_remove_A_from_B = set()
        # 创建反向依赖映射以提高效率
        reversed_dependency_map = {v: k for k, v in dependency_map.items()}
        for node_b in nodes_to_remove_B_internal:
            if node_b in reversed_dependency_map:
                nodes_to_remove_A_from_B.add(reversed_dependency_map[node_b])

        if nodes_to_remove_A_from_B:
            G1.remove_nodes_from(nodes_to_remove_A_from_B)

        # 检查是否达到稳定状态
        current_A_nodes_count = len(G1.nodes())
        current_B_nodes_count = len(G2.nodes())
        if current_A_nodes_count == last_A_nodes_count and current_B_nodes_count == last_B_nodes_count:
            break

        # 更新stage_lock逻辑
        if stage_lock:
            if len(nodes_to_remove_A) > 0 and len(nodes_to_remove_B_internal) == 0:
                stage_lock = False
            elif len(nodes_to_remove_B_internal) > 0:
                # 预测下一轮A的移除情况
                next_nodes_to_remove_A = set()
                if G1.number_of_nodes() > 0:
                    largest_cc_A_next = max(nx.connected_components(G1), key=len)
                    next_nodes_to_remove_A = set(G1.nodes()) - set(largest_cc_A_next)
                if len(next_nodes_to_remove_A) == 0:
                    stage_lock = False
                    # 既然下一轮A没有节点移除，级联基本结束，可以提前退出
                    # 将最终状态记录并退出
                    stage += 1
                    ratio_A_final = G1.number_of_nodes() / n
                    history.append({'stage': stage, 'network': 'A', 'ratio': ratio_A_final / (1 - initial_removal_fraction)})
                    print(f"阶段 {stage} 完成网络A内部失效，移除节点数: 0， 网络A剩余节点数: {G1.number_of_nodes()}，巨片存在比例为 {ratio_A_final:.4f}")
                    break

        last_A_nodes_count = current_A_nodes_count
        last_B_nodes_count = current_B_nodes_count

    return G1, G2, history


# 指定受到攻击的节点比例（选取每个阶段最大的连通分量）
def cascade_failure_max_specific_nodes(G1, G2, dependency_map, nodes_to_attack : list):
    # 复制网络以避免修改原始网络
    G1 = G1.copy()
    G2 = G2.copy()
    n = G1.number_of_nodes()

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

    # 记录每次迭代开始时节点的数量，用于判断是否达到稳定
    last_A_nodes_count = len(G1.nodes())
    last_B_nodes_count = len(G2.nodes())
    current_A_nodes_count = None
    current_B_nodes_count = None

    # 记录阶段数
    stage = 0

    # 级联失效过程
    while True:
        stage += 1
        # 阶段1: 网络A内部失效，保留最大连通分量
        if G1.number_of_nodes() > 0:
            largest_cc_A = max(nx.connected_components(G1), key=len)
            nodes_to_remove_A = set(G1.nodes()) - set(largest_cc_A)
            if nodes_to_remove_A:
                G1.remove_nodes_from(nodes_to_remove_A)
        else:
            nodes_to_remove_A = set()
        print(f"阶段 {stage} 完成网络A内部失效，移除节点数: {len(nodes_to_remove_A)}， 剩余节点数: {G1.number_of_nodes()}，巨片存在比例为 {G1.number_of_nodes()/n:.4f}")

        stage += 1
        # 阶段2: 跨网络依赖失效 (A -> B)
        nodes_to_remove_B = set(G2.nodes()) & nodes_to_remove_A # 假设节点 i in A 依赖 i in B
        if nodes_to_remove_B:
            G2.remove_nodes_from(nodes_to_remove_B)
        # 网络B内部失效，保留最大连通分量
        if G2.number_of_nodes() > 0:
            largest_cc_B = max(nx.connected_components(G2), key=len)
            nodes_to_remove_B_internal = set(G2.nodes()) - set(largest_cc_B)
            if nodes_to_remove_B_internal:
                G2.remove_nodes_from(nodes_to_remove_B_internal) 
        
        print(f"阶段 {stage} 完成网络B内部失效，移除节点数: {len(nodes_to_remove_B_internal)}， 剩余节点数: {G2.number_of_nodes()}，巨片存在比例为 {G2.number_of_nodes()/n:.4f}")

        # 保留网络B最大连通分量之后，更新网络A
        nodes_to_remove_A_from_B = set(G1.nodes()) & nodes_to_remove_B_internal # 假设节点 i in A 依赖 i in B
        if nodes_to_remove_A_from_B:
            G1.remove_nodes_from(nodes_to_remove_A_from_B)

        # 检查是否达到稳定状态
        current_A_nodes_count = len(G1.nodes())
        current_B_nodes_count = len(G2.nodes())
        if current_A_nodes_count == last_A_nodes_count and current_B_nodes_count == last_B_nodes_count:
            break

        last_A_nodes_count = current_A_nodes_count
        last_B_nodes_count = current_B_nodes_count

    return G1, G2


# 指定受到攻击的特定节点
def cascade_failure_fig2_specific_nodes(G1, G2, dependency_map, nodes_to_attack : list):
    
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

