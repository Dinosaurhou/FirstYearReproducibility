import networkx as nx
import random
from scale_free_network_generator import ScaleFreeNetworkGenerator
from scale_free_network_generator import NetworkIO


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

def create_sf_graph(_N, _average_degree, _gamma):
    # generatorSF = ScaleFreeNetworkGenerator(
    #     N=_N,
    #     avg_degree=_average_degree,
    #     gamma=_gamma,
    #     max_iterations=10000
    # )
    # G_A = generatorSF.generate()
    # G_B = generatorSF.generate()
    
    G_A = NetworkIO.load_graph("F:\\浙师大研究生\\FirstYear-Reproducibility\\FirstYearReproducibility\\scr\\byself\\saved_networks\\SFA_N30000_k4_gamma27.graphml", format='graphml')
    G_B = NetworkIO.load_graph("F:\\浙师大研究生\\FirstYear-Reproducibility\\FirstYearReproducibility\\scr\\byself\\saved_networks\\SFB_N30000_k4_gamma27.graphml", format='graphml')

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
    node_mapping = {i: i for i in range(_N)}

    return G_A, G_B, node_mapping




