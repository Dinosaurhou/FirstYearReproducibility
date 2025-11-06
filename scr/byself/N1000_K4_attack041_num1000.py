import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import CascadeFailure as cf
import numpy as np
import os
import json

rcParams['axes.unicode_minus'] = False  # 正常显示负号

if __name__ == "__main__":
    
    # 平均度 = 4，对于ER网络，p = <k>/(N-1)
    N = 2000
    average_degree = 4
    p = average_degree / (N - 1)
    
    # 设定巨片存在的标准大小
    stand_giant_size = 0.33 * N

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
        
        GA_after, GB_after = cf.cascade_failure(G_A, G_B, node_mapping, initial_removal_fraction)

        # 获取网络A中的最大连通分量
        if GA_after.number_of_nodes() > 0:
            largest_cc_A = max(nx.connected_components(GA_after), key=len)
            largest_cc_size = len(largest_cc_A)
        else:
            largest_cc_size = 0

        # 如果这个连通分量的大小>=330，就记录说明这次在此攻击比例下巨片存在
        
        if largest_cc_size >= stand_giant_size:
            exitence_count += 1
            print(f"实验 {exp_num + 1}/{num_experiments}: 巨片存在，大小为 {largest_cc_size}: 当前概率为 {exitence_count / (exp_num + 1):.4f}")
        else:
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


