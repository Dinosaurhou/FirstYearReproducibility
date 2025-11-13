import json
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline

def load_data(filename):
    """加载JSON数据文件"""
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对于脚本位置的完整文件路径
    filepath = os.path.join(script_dir, 'out', filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {filepath}")
        return None

def plot_multiple_datasets(datasets):
    """绘制多个数据集的比较图"""
    if not datasets:
        print("没有加载任何数据，无法绘图。")
        return

    # 设置中文字体，以防乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 7))

    # 定义一组标记和线型，用于区分不同的曲线
    markers = ['o', 's', '^', 'v', '>', '<', 'd']
    linestyles = ['-', '--', '-.', ':']

    # 遍历每个数据集并绘图
    for i, data in enumerate(datasets):
        if data is None:
            continue
        
        # 提取数据并转换为numpy数组
        x_attack = np.array(data['initial_removal_fractions'])
        y = np.array(data['p_giants'])
        n = data['parameters']['N']

        # 1. 将攻击比例转换为幸存节点比例
        x_survive = 1 - x_attack

        # 为了进行正确的插值，需要按x轴对数据进行排序
        sort_indices = np.argsort(x_survive)
        x_sorted = x_survive[sort_indices]
        y_sorted = y[sort_indices]

        # 2. 对曲线进行平滑处理
        # 确保有足够的数据点进行三次样条插值 (k=3)
        if len(x_sorted) > 3:
            # 创建一个更密集的x轴用于绘制平滑曲线
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            # 创建样条插值函数
            spl = make_interp_spline(x_sorted, y_sorted, k=3)
            y_smooth = spl(x_smooth)
            
            # 绘制平滑曲线和原始数据点
            line_style = linestyles[i % len(linestyles)]
            marker_style = markers[i % len(markers)]
            
            # 绘制平滑曲线（不带标记）
            plt.plot(x_smooth, y_smooth, 
                     linestyle=line_style, 
                     label=f'N = {n}')
            # 在原始数据点位置绘制标记（不带连线）
            plt.plot(x_sorted, y_sorted, 
                     marker=marker_style, 
                     linestyle='none',
                     color=plt.gca().lines[-1].get_color()) # 确保标记和曲线颜色一致
        else:
            # 如果数据点太少，直接绘制原始折线图
            plt.plot(x_sorted, y_sorted, 
                     marker=markers[i % len(markers)], 
                     linestyle=linestyles[i % len(linestyles)], 
                     label=f'N = {n}')


    # 设置图表属性
    plt.title('不同网络规模下巨片存在概率与幸存节点比例的关系', fontsize=16)
    plt.xlabel('幸存节点比例 (1-p)', fontsize=12)
    plt.ylabel('巨片存在概率 (P_giant)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 要加载的JSON文件名列表
    json_files = [
        'cf_N2000_max_data.json', 
        'cf_N4000_max_data.json',
        'cf_N8000_max_data.json',
        'cf_N16000_max_data.json',
        'cf_N32000_max_data.json'
        # 在这里添加更多文件名，例如: 'cf_N6000_max_data.json'
    ]
    # json_files = [
    #     'cf_ER_N30000_different_data.json'
    # ]

    # 加载所有数据文件
    all_data = [load_data(f) for f in json_files]
    
    # 过滤掉加载失败的数据 (结果为 None)
    loaded_data = [d for d in all_data if d is not None]

    # 绘制比较图
    if loaded_data:
        plot_multiple_datasets(loaded_data)
    else:
        print("所有数据文件加载失败，无法生成图表。")