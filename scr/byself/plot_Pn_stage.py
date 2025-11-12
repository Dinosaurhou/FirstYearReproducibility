import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_pn_stage_file(filepath):
    """
    解析P(n)与stage关系的文本文件
    返回: (N, stage_list, pn_list)
    """
    N = None
    stages = []
    pn_values = []
    
    # 正则表达式
    n_pattern = re.compile(r"N=(\d+)")
    pn_pattern = re.compile(r"P\(n=(\d+)\):\s+([\d.]+)")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取N值
                n_match = n_pattern.search(line)
                if n_match:
                    N = int(n_match.group(1))
                    continue
                
                # 提取P(n)数据
                pn_match = pn_pattern.search(line)
                if pn_match:
                    stage = int(pn_match.group(1))
                    pn = float(pn_match.group(2))
                    stages.append(stage)
                    pn_values.append(pn)
        
        if N is None:
            print(f"警告: 文件 {filepath} 中未找到N值")
            return None, None, None
        
        return N, np.array(stages), np.array(pn_values)
    
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {filepath}")
        return None, None, None

def transform_coordinates(N, stages, pn_values):
    """
    将原始数据转换为绘图坐标
    x = n * N^(-1/4)
    y = ln(P(n)) + 1/(4*ln(N))
    """
    # 过滤掉P(n)=0的点（因为ln(0)无定义）
    valid_indices = pn_values > 0
    stages_filtered = stages[valid_indices]
    pn_filtered = pn_values[valid_indices]
    
    # 计算转换后的坐标
    x = stages_filtered * (N ** (-1/4))
    # y = np.log(pn_filtered) + 1 / (4 * np.log(N))
    y = np.log(pn_filtered) + np.log(N) / 4
    
    return x, y

def plot_pn_stage_scatter():
    """
    绘制P(n)与stage关系的散点图
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件列表
    files = [
        'N1000_Pn_stage_num3000.txt',
        'N2000_Pn_stage_num3000.txt',
        'N8000_Pn_stage_num3000.txt',
        'N16000_Pn_stage_num3000.txt',
        'N32000_Pn_stage_num3000.txt',
        'N64000_Pn_stage_num3000.txt',
    ]
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义不同的标记样式和颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 遍历每个文件
    for idx, filename in enumerate(files):
        filepath = os.path.join(script_dir, 'out', filename)
        
        # 解析文件
        N, stages, pn_values = parse_pn_stage_file(filepath)
        
        if N is None:
            continue
        
        # 转换坐标
        x, y = transform_coordinates(N, stages, pn_values)
        
        # 绘制散点图
        ax.scatter(x, y, 
                  marker=markers[idx % len(markers)],
                  color=colors[idx % len(colors)],
                  s=50,
                  alpha=0.7,
                  label=f'N = {N}',
                  edgecolors='black',
                  linewidths=0.5)
        
        print(f"已绘制 N={N} 的数据，共 {len(x)} 个点")
    
    # 设置图表属性
    ax.set_xlabel(r'$n \cdot N^{-1/4}$', fontsize=14)
    ax.set_ylabel(r'$\ln(P(n)) + \frac{\ln(N)}{4}$', fontsize=14)
    # ax.set_title('级联失效阶段数分布的标度分析', fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_pn_stage_scatter()