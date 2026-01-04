import json
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d  # 引入高斯滤波
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

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 7))

    # 定义颜色列表，确保不同N值颜色区分明显
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 遍历每个数据集并绘图
    for i, data in enumerate(datasets):
        if data is None:
            continue
        
        # 提取数据
        x_attack = np.array(data['initial_removal_fractions'])
        y = np.array(data['p_giants'])
        n = data['parameters']['N']

        # 1. 转换x轴数据
        x_survive = (1 - x_attack) * 4

        # 排序
        sort_indices = np.argsort(x_survive)
        x_sorted = x_survive[sort_indices]
        y_sorted = y[sort_indices]

        # --- 核心修改：两步平滑法 ---
        
        # 第一步：高斯滤波 (去除突出的噪点)
        # sigma 控制平滑程度，数值越大越平滑，建议 0.8 - 1.5 之间
        y_denoised = gaussian_filter1d(y_sorted, sigma=1.0)

        # 第二步：B-样条插值 (增加点数，使曲线圆滑)
        if len(x_sorted) > 3:
            # 生成密集的x轴
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
            
            # 使用去噪后的数据进行插值
            # k=3 (三次样条), bc_type='natural' (自然边界条件，防止端点飞出)
            spl = make_interp_spline(x_sorted, y_denoised, k=3)
            y_smooth = spl(x_smooth)
            
            # 限制y值范围在[0, 1]之间（物理意义约束）
            y_smooth = np.clip(y_smooth, 0, 1)
            
            # 绘图
            color = colors[i % len(colors)]
            plt.plot(x_smooth, y_smooth, 
                     linestyle='-', 
                     linewidth=2.5, 
                     color=color,
                     label=f'N = {n}')
            
            # 可选：绘制原始点（半透明），用于对比
            # plt.scatter(x_sorted, y_sorted, color=color, alpha=0.3, s=20)
            
        else:
            # 数据点太少时的回退方案
            plt.plot(x_sorted, y_sorted, label=f'N = {n}')

    # 添加理论临界线
    critical_point = 2.4554
    plt.axvline(x=critical_point, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.text(critical_point + 0.01, 0.5, f'理论临界点\n{critical_point}', color='gray', fontsize=10)

    # 设置图表属性
    plt.title('不同网络规模下巨片存在概率与平均度的关系', fontsize=16)
    plt.xlabel('平均度 <k>', fontsize=12)
    plt.ylabel('巨片存在概率 (P_giant)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=10)

    # --- 修改开始：强制坐标轴从左下角开始 ---
    # 1. 获取当前数据的 x 轴最小值（或者你可以手动指定为 0 或 2.3 等）
    # 这里假设你想从 x 的最小值开始，或者从 2.3 开始（根据你的数据范围）
    # 如果你想从 0 开始，就写 plt.xlim(left=0)
    # 根据你的数据，x 轴大约在 2.3 到 2.5 之间，建议根据数据自动调整或手动设定
    
    # 示例：强制 Y 轴从 0 开始
    plt.ylim(bottom=0, top=1.05) 

    # 示例：强制 X 轴从数据范围的左边界开始 (或者你可以写 left=2.3)
    # 为了紧贴左下角，我们需要知道 x 的最小值。
    # 我们可以遍历所有数据找到最小的 x_smooth
    min_x = 100
    max_x = 0
    for data in datasets:
        if data:
            x_survive = (1 - np.array(data['initial_removal_fractions'])) * 4
            min_x = min(min_x, x_survive.min())
            max_x = max(max_x, x_survive.max())
            
    plt.xlim(left=min_x, right=max_x)

    # 2. 消除坐标轴与数据之间的留白
    plt.margins(x=0, y=0)
    # --- 修改结束 ---

    plt.tight_layout()

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

    # 加载所有数据文件
    all_data = [load_data(f) for f in json_files]
    
    # 过滤掉加载失败的数据 (结果为 None)
    loaded_data = [d for d in all_data if d is not None]

    # 绘制比较图
    if loaded_data:
        plot_multiple_datasets(loaded_data)
    else:
        print("所有数据文件加载失败，无法生成图表。")