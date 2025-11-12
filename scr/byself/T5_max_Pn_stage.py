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
    N = 32000
    average_degree = 4

    G_A, G_B, node_mapping = create_er_graph(N, average_degree)

    # 进行多次实验，改变初始攻击比例(1 - 2.4554 / 4)
    initial_removal_fraction = 1 - 2.4554 / average_degree
    
    p_giants = []
    num_experiments = 3000 # 假设实验3000次

    print(f"\n{'='*50}")
    print(f"初始攻击比例: {initial_removal_fraction:.4f}")
    print(f"总实验次数: {num_experiments}")
    print(f"{'='*50}")

    exitence_count = 0  # 记录巨片存在的次数
    final_stage_counts = {} # 新增：用于统计每个最终阶段数出现的次数
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    for exp_num in range(num_experiments):

        # G_A, G_B, node_mapping = create_er_graph(N, average_degree)
        
        GA_after, GB_after, final_history = cf.cascade_failure_max_change_stagecount(G_A, G_B, node_mapping, initial_removal_fraction)

        # --- 新增：统计最终阶段数 ---
        if final_history:
            # 获取本次实验的最终阶段数
            final_stage = final_history[-1]['stage']
            # 对该阶段数进行计数
            final_stage_counts[final_stage] = final_stage_counts.get(final_stage, 0) + 1
        # --- 统计结束 ---

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
    print(f"\n攻击比例：{initial_removal_fraction:.4f} -> 巨片存在概率：{p_giant:.4f}")

    # --- 新增：计算并打印P(n) ---
    print(f"N={N}")
    print(f"num_experiments={num_experiments}")
    print(f"\n{'='*50}")
    print("级联失效结束阶段概率 P(n):")
    print(f"{'='*50}")
    
    # 将字典按阶段数排序，以便清晰显示
    sorted_stages = sorted(final_stage_counts.keys())
    
    stage_probabilities = {}
    for stage in sorted_stages:
        count = final_stage_counts[stage]
        probability = count / num_experiments
        stage_probabilities[stage] = probability
        print(f"P(n={stage}): {probability:.4f}  ({count}/{num_experiments} 次)")
    
    # --- P(n) 计算和打印结束 ---

    # --- 新增：绘制 P(n) 关系图 ---
    
    # 准备绘图数据
    # 过滤掉概率为0的阶段，因为log(0)是未定义的
    plot_stages = [s for s in sorted_stages if stage_probabilities.get(s, 0) > 0]
    
    if plot_stages:
        # 计算变换后的坐标
        x_transformed = [n * (N ** (-1/4)) for n in plot_stages]
        y_transformed = [np.log(stage_probabilities[n]) + np.log(N) / 4 for n in plot_stages]

        # 创建新图形
        plt.figure(figsize=(10, 6))
        
        # 绘制散点图
        plt.scatter(x_transformed, y_transformed, label='实验数据点', color='blue', alpha=0.7)

        # 设置图表标题和坐标轴标签
        plt.title('级联失效结束阶段的变换概率分布')
        plt.xlabel('n * N^(-1/4)')
        plt.ylabel('ln(P(n)) + ln(N)/4')
        
        # 添加网格和图例
        plt.grid(True)
        plt.legend()
        
        # 显示图形
        plt.show()
    else:
        print("\n没有有效的阶段数据可用于绘图。")
    # --- 绘图结束 ---


    # 输出结果汇总
    print(f"\n{'='*50}")
    print("实验结果汇总:")
    print(f"{'='*50}")

    for frac, p_g in zip([initial_removal_fraction], p_giants):
        print(f"初始攻击比例: {frac:.4f} -> 巨片存在概率: {p_g:.4f}")




