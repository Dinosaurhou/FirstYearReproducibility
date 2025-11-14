import json
import matplotlib.pyplot as plt
import os
import numpy as np

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
        
        # 获取网络类型作为图例标签
        network_type = data['parameters'].get('network_type', '未知网络类型')

        # 将攻击比例转换为幸存节点比例
        x_survive = 1 - x_attack

        # 按x轴对数据进行排序
        sort_indices = np.argsort(x_survive)
        x_sorted = x_survive[sort_indices]
        y_sorted = y[sort_indices]

        # 绘制曲线
        line_style = linestyles[i % len(linestyles)]
        marker_style = markers[i % len(markers)]
        
        plt.plot(x_sorted, y_sorted, 
                 marker=marker_style, 
                 linestyle=line_style, 
                 linewidth=2,
                 markersize=6,
                 label=f'{network_type}')

    # 设置图表属性
    plt.title('不同网络类型下巨片存在概率与幸存节点比例的关系', fontsize=16, fontweight='bold')
    plt.xlabel('幸存节点比例 (1-p)', fontsize=14)
    plt.ylabel('巨片存在概率 (P_giant)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 要加载的JSON文件名列表
    json_files = [
        'cf_ER_N30000_data.json',
        'cf_RR_N30000_data.json'
    ]

    # 加载所有数据文件
    all_data = [load_data(f) for f in json_files]
    
    # 过滤掉加载失败的数据 (结果为 None)
    loaded_data = [d for d in all_data if d is not None]

    # 绘制比较图
    if loaded_data:
        plot_multiple_datasets(loaded_data)
    else:
        print("所有数据文件加载失败，无法生成图表。")