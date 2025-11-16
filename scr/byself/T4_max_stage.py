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


if __name__ == "__main__":

# 平均度 = 4，对于ER网络，p = <k>/(N-1)
    N = 16000
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


    # 进行多次实验，改变初始攻击比例(1 - 2.45 / 4)
    initial_removal_fraction = 0.3875
    
    p_giants = []
    num_experiments = 50 # 每个初始攻击比例重复50次

    print(f"\n{'='*50}")
    print(f"初始攻击比例: {initial_removal_fraction:.4f}")
    print(f"{'='*50}")

    exitence_count = 0  # 记录巨片存在的次数
    
    # --- 新增：初始化绘图 ---
    fig, ax = plt.subplots(figsize=(12, 8))
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for exp_num in range(num_experiments):
        
        GA_after, GB_after, final_history = cf.cascade_failure_max_change_stagecount(G_A, G_B, node_mapping, initial_removal_fraction)

        # --- 新增：处理并绘制当前实验的历史数据 ---
        if final_history:
            stages = np.array([item['stage'] for item in final_history])
            ratios = np.array([item['ratio'] for item in final_history])
            
            # 为了平滑曲线，需要确保有足够的数据点
            if len(stages) > 2:
                # 创建更密集的stage轴用于平滑插值
                stages_smooth = np.linspace(stages.min(), stages.max(), 300)
                # 创建样条插值函数
                spl = make_interp_spline(stages, ratios, k=3)  # k=3表示三次样条插值
                ratios_smooth = spl(stages_smooth)
                # 绘制平滑曲线，设置透明度以便观察重叠
                ax.plot(stages_smooth, ratios_smooth, alpha=0.3, linewidth=1.5)
            else:
                # 如果数据点太少，直接绘制折线
                ax.plot(stages, ratios, alpha=0.3, linewidth=1.5)


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
    
    # --- 新增：完成绘图并显示 ---
    ax.set_title(f'级联失效过程中巨片比例的变化 (攻击比例 p={initial_removal_fraction})', fontsize=16)
    ax.set_xlabel('失效阶段 (Stage)', fontsize=12)
    ax.set_ylabel('巨片相对大小 (P∞(p,t) / P∞(p,0))', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


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




