import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import CascadeFailure as cf
import GenerateNetwork as gn
import os
import json

rcParams['axes.unicode_minus'] = False  # 正常显示负号

if __name__ == "__main__":

    N = 30000
    average_degree = 4

    # 进行多次实验，改变初始攻击比例，λ=3
    # initial_removal_fractions = np.linspace(1 - 0.7125, 1 - 0.55, 40)
    # λ=2.3
    initial_removal_fractions = np.linspace(1 - 0.78625, 1 - 0.65, 50)
    # λ=2.7
    # initial_removal_fractions = np.linspace(1 - 0.75, 1 - 0.645, 50)

    # initial_removal_fractions = list([0.9])

    p_giants = []
    num_experiments = 300 # 每个初始攻击比例重复300次

    # G_A, G_B, node_mapping = gn.create_rr_graph(N, average_degree)
    # G_A, G_B, node_mapping = gn.create_sf_graph(N, average_degree, 3.0)
    G_A, G_B, node_mapping = gn.create_sf_graph(N, average_degree, 2.3)

    for initial_removal_fraction in initial_removal_fractions:

        print(f"\n{'='*50}")
        print(f"初始攻击比例: {initial_removal_fraction:.4f}")
        print(f"{'='*50}")

        exitence_count = 0  # 记录巨片存在的次数
        
        for exp_num in range(num_experiments):
            
            # 创建两个ER随机图和节点映射
            # G_A, G_B, node_mapping = create_er_graph(N, average_degree)

            GA_after, GB_after, history= cf.cascade_failure_max_change_stagecount(G_A, G_B, node_mapping, initial_removal_fraction)

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

    for frac, p_g in zip(initial_removal_fractions, p_giants):
        print(f"初始攻击比例: {frac:.4f} -> 巨片存在概率: {p_g:.4f}")

    # XXX:保存数据到文件
    # 确保目录存在
    save_dir = "./out"
    os.makedirs(save_dir, exist_ok=True)

    # 保存数据为JSON格式
    data = {
        'initial_removal_fractions': initial_removal_fractions.tolist(),
        'p_giants': p_giants,
        'parameters': {
            'N': N,
            'average_degree': average_degree,
            'num_experiments': num_experiments,
            'network_type': 'SF2.3'
        }
    }

    file_path = os.path.join(save_dir, 'cf_New_SF2.3_N' + str(N) + '_data.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n数据已保存到: {file_path}")

    # 绘制巨片存在概率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(initial_removal_fractions, p_giants, marker='o')
    plt.xlabel("初始攻击比例")
    plt.ylabel("巨片存在概率")
    plt.title("巨片存在概率与初始攻击比例的关系")
    plt.grid()
    plt.show()




