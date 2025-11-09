import json
import matplotlib.pyplot as plt
import os

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

def plot_comparison(data1, data2):
    """绘制两个数据集的比较图"""
    if data1 is None or data2 is None:
        print("数据加载失败，无法绘图。")
        return

    # 设置中文字体，以防乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 7))

    # 提取数据
    x1 = data1['initial_removal_fractions']
    y1 = data1['p_giants']
    n1 = data1['parameters']['N']

    x2 = data2['initial_removal_fractions']
    y2 = data2['p_giants']
    n2 = data2['parameters']['N']

    # 绘图
    plt.plot(x1, y1, marker='o', linestyle='-', label=f'N = {n1}')
    plt.plot(x2, y2, marker='s', linestyle='--', label=f'N = {n2}')

    # 设置图表属性
    plt.title('不同网络规模下巨片存在概率与攻击比例的关系', fontsize=16)
    plt.xlabel('受攻击节点比例 (p)', fontsize=12)
    plt.ylabel('巨片存在概率 (P_giant)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 加载数据
    data_n1000 = load_data('cf_N1000_stand_giant.json')
    data_n2000 = load_data('cf_N2000_stand_giant.json')

    # 绘制比较图
    plot_comparison(data_n1000, data_n2000)