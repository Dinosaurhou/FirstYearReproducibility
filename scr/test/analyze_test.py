import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_network(graphml_file):
    """
    分析 GraphML 网络文件的基本性质
    
    参数:
        graphml_file: GraphML 文件路径
    """
    # 读取网络
    print(f"正在读取网络文件: {graphml_file}")
    G = nx.read_graphml(graphml_file)
    
    # 将节点标签转换为整数（GraphML 可能存储为字符串）
    G = nx.convert_node_labels_to_integers(G)
    
    # 基本统计信息
    print("\n" + "="*60)
    print("网络基本信息")
    print("="*60)
    print(f"节点数 (N): {G.number_of_nodes()}")
    print(f"边数 (E): {G.number_of_edges()}")
    
    # 计算平均度
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    print(f"平均度 <k>: {avg_degree:.4f}")
    print(f"最大度: {max(degrees)}")
    print(f"最小度: {min(degrees)}")
    
    # 连通性
    is_connected = nx.is_connected(G)
    print(f"是否连通: {is_connected}")
    if not is_connected:
        num_components = nx.number_connected_components(G)
        print(f"连通分量数: {num_components}")
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"最大连通分量大小: {len(largest_cc)} ({len(largest_cc)/G.number_of_nodes()*100:.2f}%)")
    
    # 聚类系数
    avg_clustering = nx.average_clustering(G)
    print(f"平均聚类系数: {avg_clustering:.4f}")
    
    # 自环和重边检查
    num_selfloops = nx.number_of_selfloops(G)
    print(f"自环数: {num_selfloops}")
    
    print("="*60)
    
    return G, degrees

def plot_degree_distribution(degrees, gamma=2.7, save_path=None):
    """
    绘制度分布的对数坐标散点图，并叠加理论幂律曲线
    
    参数:
        degrees: 度序列
        gamma: 幂律指数
        save_path: 保存路径（可选）
    """
    # 统计度分布
    degree_counts = np.bincount(degrees)
    k_values = np.arange(len(degree_counts))
    
    # 计算概率
    total_nodes = len(degrees)
    pk_values = degree_counts / total_nodes
    
    # 过滤掉度为0的节点
    mask = (pk_values > 0) & (k_values > 0)
    k_plot = k_values[mask]
    pk_plot = pk_values[mask]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # === 左图：对数坐标 ===
    # 绘制实际数据散点
    ax1.loglog(k_plot, pk_plot, 'o', markersize=8, alpha=0.7, 
               label='实际数据', color='#2E86DE')
    
    # 绘制理论幂律曲线
    k_min = k_plot.min()
    k_max = k_plot.max()
    k_theory = np.logspace(np.log10(k_min), np.log10(k_max), 100)
    
    # 归一化：使理论曲线与实际数据对齐
    # 使用最小二乘法找到最佳归一化常数
    log_k_data = np.log(k_plot)
    log_pk_data = np.log(pk_plot)
    # 拟合 log(P(k)) = C - gamma * log(k)
    # 使用中间区域的数据点进行拟合
    mid_start = len(k_plot) // 4
    mid_end = 3 * len(k_plot) // 4
    C_fit = np.mean(log_pk_data[mid_start:mid_end] + gamma * log_k_data[mid_start:mid_end])
    
    pk_theory = np.exp(C_fit) * (k_theory ** (-gamma))
    
    ax1.loglog(k_theory, pk_theory, '--', linewidth=2.5, 
               label=f'理论: $P(k) \\propto k^{{-{gamma}}}$', color='#EE5A24')
    
    ax1.set_xlabel('度 k', fontsize=13, fontweight='bold')
    ax1.set_ylabel('概率 P(k)', fontsize=13, fontweight='bold')
    ax1.set_title('度分布 (对数坐标)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
    ax1.legend(fontsize=11, loc='best')
    ax1.tick_params(labelsize=11)
    
    # === 右图：线性坐标 ===
    ax2.scatter(k_plot, pk_plot, s=50, alpha=0.6, 
                label='实际数据', color='#2E86DE')
    ax2.plot(k_theory, pk_theory, '--', linewidth=2.5, 
             label=f'理论: $k^{{-{gamma}}}$', color='#EE5A24')
    
    ax2.set_xlabel('度 k', fontsize=13, fontweight='bold')
    ax2.set_ylabel('概率 P(k)', fontsize=13, fontweight='bold')
    ax2.set_title('度分布 (线性坐标)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.legend(fontsize=11, loc='best')
    ax2.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 图像已保存到: {save_path}")
    
    plt.show()

def estimate_gamma(degrees):
    """
    使用最大似然估计 (MLE) 估计幂律指数
    
    参数:
        degrees: 度序列
        
    返回:
        gamma_mle: 估计的幂律指数
    """
    degrees = np.array(degrees)
    k_min = max(1, int(np.percentile(degrees, 25)))
    
    degrees_filtered = degrees[degrees >= k_min]
    
    if len(degrees_filtered) < 10:
        return None
    
    n = len(degrees_filtered)
    log_ratio_sum = np.sum(np.log(degrees_filtered / k_min))
    
    if log_ratio_sum > 0:
        gamma_mle = 1 + n / log_ratio_sum
        return gamma_mle
    else:
        return None

if __name__ == "__main__":
    # 设置文件路径
    graphml_file = Path("../byself/saved_networks/SFA_N30000_k4_gamma3.graphml")
    
    # 检查文件是否存在
    if not graphml_file.exists():
        print(f"错误: 文件不存在 - {graphml_file}")
        print("请检查文件路径是否正确")
    else:
        # 分析网络
        G, degrees = analyze_network(graphml_file)
        
        # 估计幂律指数
        gamma_estimated = estimate_gamma(degrees)
        if gamma_estimated:
            print(f"\n估计的幂律指数 γ (MLE): {gamma_estimated:.4f}")
        
        # 绘制度分布图
        print("\n正在生成度分布图...")
        plot_degree_distribution(degrees, gamma=2.7, 
                                save_path='network_degree_distribution_gamma27.png')