import networkx as nx
import random

def create_er_graph(N, average_degree):
    """创建一个Erdos-Renyi随机图"""
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

    return G_A, G_B, node_mapping

def create_rr_graph(N, average_degree):
    """创建一个随机正则图"""
    k = average_degree
    # 创建随机正则图 - 网络A
    G_A = nx.random_regular_graph(k, N)
    # 创建随机正则图 - 网络B（使用不同的随机种子）
    G_B = nx.random_regular_graph(k, N)

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

    return G_A, G_B, node_mapping

def create_sf_graph(N, average_degree, gamma=3, max_tries=100):
    """创建一个无标度网络(Scale-Free network)
    
    参数:
        N: 节点数
        average_degree: 目标平均度
        gamma: 幂律度分布指数,默认为3
        max_tries: 最大尝试次数
    """
    # 特殊情况：gamma = 3 时使用 BA 模型
    if abs(gamma - 3.0) < 1e-6:
        m = int(average_degree / 2)
        if m < 1:
            m = 1
        
        # 创建网络A - 使用BA模型
        G_A = nx.barabasi_albert_graph(N, m)
        # 移除可能的自环
        G_A = nx.Graph(G_A)
        G_A.remove_edges_from(nx.selfloop_edges(G_A))
        
        # 创建网络B - 使用BA模型
        G_B = nx.barabasi_albert_graph(N, m)
        G_B = nx.Graph(G_B)
        G_B.remove_edges_from(nx.selfloop_edges(G_B))
        
        # 打印网络A基本信息
        print("=== 网络A (SF-BA) ===")
        print(f"节点数: {G_A.number_of_nodes()}")
        print(f"边数: {G_A.number_of_edges()}")
        print(f"实际平均度: {2 * G_A.number_of_edges() / G_A.number_of_nodes():.2f}")
        
        # 打印网络B基本信息
        print("\n=== 网络B (SF-BA) ===")
        print(f"节点数: {G_B.number_of_nodes()}")
        print(f"边数: {G_B.number_of_edges()}")
        print(f"实际平均度: {2 * G_B.number_of_edges() / G_B.number_of_nodes():.2f}")
        
        # 创建随机节点对应关系
        nodes_A = list(range(N))
        nodes_B = list(range(N))
        random.shuffle(nodes_B)  # 打乱网络B的节点顺序
        node_mapping = {nodes_A[i]: nodes_B[i] for i in range(N)}
        
        return G_A, G_B, node_mapping
    
    # gamma != 3 时使用 Configuration Model
    # for attempt in range(max_tries):
    n = 0
    while True:
        # 使用configuration model生成幂律度分布的网络
        # 生成度序列
        n += 1
        print(f"尝试第{n}次")
        degree_sequence = nx.utils.powerlaw_sequence(N, gamma)
        degree_sequence = [int(d) for d in degree_sequence]
        
        # 确保度序列之和为偶数(图论要求)
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[0] += 1
        
        # 确保度序列中没有超过N-1的度(避免自环和多重边)
        degree_sequence = [min(d, N-1) for d in degree_sequence]
        
        try:
            # 创建网络A - 使用configuration_model
            G_A = nx.configuration_model(degree_sequence)
            # 移除自环和多重边
            G_A = nx.Graph(G_A)
            G_A.remove_edges_from(nx.selfloop_edges(G_A))
            
            # 创建网络B
            G_B = nx.configuration_model(degree_sequence)
            G_B = nx.Graph(G_B)
            G_B.remove_edges_from(nx.selfloop_edges(G_B))
            
            # 检查节点数是否正确
            if G_A.number_of_nodes() == N and G_B.number_of_nodes() == N:
                # 计算实际平均度
                actual_avg_A = 2 * G_A.number_of_edges() / N
                actual_avg_B = 2 * G_B.number_of_edges() / N
                
                # 如果平均度接近目标值,则接受
                if abs(actual_avg_A - average_degree) < average_degree * 0.08 and abs(actual_avg_B - average_degree) < average_degree * 0.08:
                    break
        except:
            continue
    
    # 打印网络A基本信息
    print("=== 网络A (SF) ===")
    print(f"节点数: {G_A.number_of_nodes()}")
    print(f"边数: {G_A.number_of_edges()}")
    print(f"实际平均度: {2 * G_A.number_of_edges() / G_A.number_of_nodes():.2f}")
    
    # 打印网络B基本信息
    print("\n=== 网络B (SF) ===")
    print(f"节点数: {G_B.number_of_nodes()}")
    print(f"边数: {G_B.number_of_edges()}")
    print(f"实际平均度: {2 * G_B.number_of_edges() / G_B.number_of_nodes():.2f}")
    
    # 创建随机节点对应关系
    nodes_A = list(range(N))
    nodes_B = list(range(N))
    random.shuffle(nodes_B)  # 打乱网络B的节点顺序
    node_mapping = {nodes_A[i]: nodes_B[i] for i in range(N)}
    
    return G_A, G_B, node_mapping





