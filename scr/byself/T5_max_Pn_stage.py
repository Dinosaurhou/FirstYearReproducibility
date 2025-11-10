import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import CascadeFailure as cf
import os
import json
from scipy.interpolate import make_interp_spline


rcParams['axes.unicode_minus'] = False  # 正常显示负号

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

if __name__ == "__main__":

    # 平均度 = 4，对于ER网络，p = <k>/(N-1)
    N = 8000
    average_degree = 4

    G_A, G_B, node_mapping = create_er_graph(N, average_degree)

    # 进行多次实验，改变初始攻击比例(1 - 2.4554 / 4)
    initial_removal_fraction = 1 - 2.4554 / average_degree
    
    p_giants = []
    num_experiments = 50 # 单个网络实验重复50次

    print(f"\n{'='*50}")
    print(f"初始攻击比例: {initial_removal_fraction:.4f}")
    print(f"{'='*50}")

    exitence_count = 0  # 记录巨片存在的次数
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for exp_num in range(num_experiments):
        
        GA_after, GB_after, final_history = cf.cascade_failure_max(G_A, G_B, node_mapping, initial_removal_fraction)

        

        # 获取网络A中的最大连通分量
        if GA_after.number_of_nodes() > 0:
            largest_cc_A = max(nx.connected_components(GA_after), key=len)
            largest_cc_size = len(largest_cc_A)
        else:
            largest_cc_size = 0

        # 如果这个连通分量的大小>=10，就记录说明这次在此攻击比例下巨片存在，记录+1
        if largest_cc_size >= 10:
            exitence_count += 1
            print(f"实验 {exp_num + 1}/{num_experiments}: 巨片存在，大小为 {largest_cc_size}: 当前概率为 {exitence_count / (exp_num + 1):.4f}")
        else:
            print(f"实验 {exp_num + 1}/{num_experiments}: 巨片不存在，大小为 {largest_cc_size}")
    
    

    # 计算巨片存在的概率
    p_giant = exitence_count / num_experiments
    p_giants.append(p_giant)
    print(f"攻击比例：{initial_removal_fraction:.4f} -> 巨片存在概率：{p_giant:.4f}")

    # 输出结果汇总
    print(f"\n{'='*50}")
    print("实验结果汇总:")
    print(f"{'='*50}")

    for frac, p_g in zip([initial_removal_fraction], p_giants):
        print(f"初始攻击比例: {frac:.4f} -> 巨片存在概率: {p_g:.4f}")




