import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from typing import Tuple, Optional
import warnings
import pickle
import json
import networkx as nx
from pathlib import Path

class NetworkIO:
    """网络文件输入/输出工具类"""
    
    @staticmethod
    def save_graph(G: nx.Graph, filepath: str, format: str = 'graphml', 
                   include_metadata: bool = True):
        """
        保存网络到文件
        
        参数:
            G: NetworkX图对象
            filepath: 保存路径
            format: 保存格式，支持 'gpickle', 'edgelist', 'gml', 'graphml'
            include_metadata: 是否保存元数据（节点数、边数等）
        """
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'gml':
            # 方式3: GML格式（通用格式，可被其他软件读取）
            nx.write_gml(G, filepath)
            print(f"✓ 网络已保存为GML格式: {filepath}")
            
        elif format == 'graphml':
            # 方式4: GraphML格式（XML格式，功能强大）
            nx.write_graphml(G, filepath)
            print(f"✓ 网络已保存为GraphML格式: {filepath}")
            
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        # 保存元数据
        if include_metadata:
            metadata_path = filepath + '.meta.json'
            metadata = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
                'is_connected': nx.is_connected(G),
                'num_components': nx.number_connected_components(G),
                'self_loops': nx.number_of_selfloops(G)
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ 元数据已保存: {metadata_path}")
    
    @staticmethod
    def load_graph(filepath: str, format: str = 'graphml') -> nx.Graph:
        """
        从文件加载网络
        
        参数:
            filepath: 文件路径
            format: 文件格式
            
        返回:
            NetworkX图对象
        """
        if format == 'gml':
            G = nx.read_gml(filepath)
            print(f"✓ 从GML格式加载网络: {filepath}")
            
        elif format == 'graphml':
            G = nx.read_graphml(filepath)
            # GraphML可能将节点ID读取为字符串，需要转换
            G = nx.convert_node_labels_to_integers(G)
            print(f"✓ 从GraphML格式加载网络: {filepath}")
            
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        # 加载元数据（如果存在）
        metadata_path = filepath + '.meta.json'
        if Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✓ 元数据已加载:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        return G

class ScaleFreeNetworkGenerator:
    """
    无标度网络生成器
    
    功能：生成具有指定参数的无标度网络
    - 可自定义节点数N、平均度<k>、幂律指数λ
    - 无环路、无重边
    - 允许孤立节点
    - 严格控制节点数和平均度误差
    """
    
    def __init__(self, N: int, avg_degree: float, gamma: float, 
                 max_iterations: int = 100, random_seed: Optional[int] = None):
        """
        初始化网络生成器
        
        参数:
            N: 节点数
            avg_degree: 平均度 <k>
            gamma: 幂律指数 λ (通常在2到3之间)
            max_iterations: 最大尝试次数
            random_seed: 随机种子
        """
        self.N = N
        self.avg_degree = avg_degree
        self.gamma = gamma
        self.max_iterations = max_iterations
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 验证参数
        if not (2 <= gamma <= 3):
            warnings.warn(f"幂律指数γ={gamma}不在推荐范围[2,3]内")
        
        if avg_degree >= N - 1:
            raise ValueError(f"平均度{avg_degree}必须小于N-1={N-1}")
    
    def generate_degree_sequence(self) -> np.ndarray:
        """
        使用幂律分布生成度序列
        
        返回:
            度序列数组
        """
        # 最小度设为1，最大度限制为N-1
        k_min = 1
        k_max = min(int(self.N * 0.5), self.N - 1)  # 限制最大度避免过大
        
        # 生成幂律分布的度序列
        # P(k) ∝ k^(-γ)
        degrees = []
        
        # 归一化常数
        C = sum([k**(-self.gamma) for k in range(k_min, k_max + 1)])
        
        for _ in range(self.N):
            # 使用逆变换采样
            r = np.random.random()
            cumsum = 0
            for k in range(k_min, k_max + 1):
                cumsum += k**(-self.gamma) / C
                if r <= cumsum:
                    degrees.append(k)
                    break
            else:
                degrees.append(k_max)
        
        degrees = np.array(degrees)
        
        # 调整度序列使总度数为偶数（图论要求）
        total_degree = np.sum(degrees)
        if total_degree % 2 != 0:
            # 随机选择一个节点增加度数
            idx = np.random.randint(0, self.N)
            if degrees[idx] < k_max:
                degrees[idx] += 1
            else:
                # 如果已经是最大度，找一个非最大度的节点
                non_max = np.where(degrees < k_max)[0]
                if len(non_max) > 0:
                    degrees[non_max[0]] += 1
                else:
                    degrees[idx] -= 1  # 降低而不是增加
        
        return degrees
    
    def adjust_degree_sequence(self, degrees: np.ndarray) -> np.ndarray:
        """
        调整度序列以匹配目标平均度
        
        参数:
            degrees: 原始度序列
            
        返回:
            调整后的度序列
        """
        current_avg = np.mean(degrees)
        target_avg = self.avg_degree
        
        # 缩放因子
        scale_factor = target_avg / current_avg
        
        # 缩放度序列
        adjusted_degrees = degrees * scale_factor
        adjusted_degrees = np.round(adjusted_degrees).astype(int)
        
        # 确保最小度为0（允许孤立节点），最大度不超过N-1
        adjusted_degrees = np.clip(adjusted_degrees, 0, self.N - 1)
        
        # 调整总度数为偶数
        total_degree = np.sum(adjusted_degrees)
        if total_degree % 2 != 0:
            # 找一个度数小于N-1的节点增加度数
            candidates = np.where(adjusted_degrees < self.N - 1)[0]
            if len(candidates) > 0:
                idx = np.random.choice(candidates)
                adjusted_degrees[idx] += 1
            else:
                # 否则减少一个度数最大的节点的度数
                idx = np.argmax(adjusted_degrees)
                adjusted_degrees[idx] -= 1
        
        return adjusted_degrees
    
    def configuration_model(self, degrees: np.ndarray) -> nx.Graph:
        """
        使用配置模型生成网络（改进版，严格避免自环和重边）
        
        参数:
            degrees: 度序列
            
        返回:
            生成的网络图
        """
        # 创建存根列表
        stubs = []
        for node, degree in enumerate(degrees):
            stubs.extend([node] * degree)
        
        # 随机配对，多次尝试避免自环和重边
        max_attempts = 50
        best_G = None
        best_edges = 0
        
        for attempt in range(max_attempts):
            # 随机打乱
            np.random.shuffle(stubs)
            
            G = nx.Graph()
            G.add_nodes_from(range(self.N))
            
            # 配对创建边
            for i in range(0, len(stubs) - 1, 2):
                u, v = stubs[i], stubs[i + 1]
                # 避免自环和重边
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
            
            # 记录最佳结果
            if G.number_of_edges() > best_edges:
                best_edges = G.number_of_edges()
                best_G = G.copy()
            
            # 如果边数接近理论值，提前退出
            expected_edges = sum(degrees) // 2
            if G.number_of_edges() >= expected_edges * 0.95:
                best_G = G
                break
        
        # 最终清理：确保没有自环和重边
        if best_G is not None:
            # 移除自环
            self_loops = list(nx.selfloop_edges(best_G))
            best_G.remove_edges_from(self_loops)
            
            # NetworkX的Graph类自动避免重边，但再次确认
            # 转换为简单图（无自环、无重边）
            best_G = nx.Graph(best_G)
        
        return best_G if best_G is not None else nx.Graph()
    
    def _ensure_simple_graph(self, G: nx.Graph) -> nx.Graph:
        """确保图是简单图（无自环、无重边）"""
        # 移除所有自环
        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            G.remove_edges_from(self_loops)
            print(f"  警告: 移除了 {len(self_loops)} 个自环")
        
        # NetworkX的Graph类自动处理重边（不会添加重复边）
        # 但为了确保，我们转换为简单图
        G_simple = nx.Graph(G)
        
        # 验证
        assert nx.number_of_selfloops(G_simple) == 0, "仍存在自环！"
        
        return G_simple
    
    def _generate_ba_network(self) -> nx.Graph:
        """
        使用 Barabási-Albert (BA) 模型生成网络
        
        BA 模型对应 γ = 3 的情况
        
        返回:
            生成的BA网络
        """
        # BA模型参数：m = 每个新节点连接的边数
        # 平均度 ≈ 2m，所以 m ≈ avg_degree / 2
        m = max(1, int(self.avg_degree / 2))
        
        print(f"  使用 BA 模型参数: m = {m}")
        
        # 生成BA网络
        G = nx.barabasi_albert_graph(self.N, m, np.random)
        
        # 转换为简单图（移除可能的自环，虽然BA模型理论上不会产生自环）
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # 计算实际统计量
        actual_degrees = [G.degree(n) for n in G.nodes()]
        actual_avg_degree = np.mean(actual_degrees)
        avg_degree_error = abs(actual_avg_degree - self.avg_degree)
        
        print(f"✓ BA 网络生成成功")
        print(f"  节点数: {G.number_of_nodes()} (严格)")
        print(f"  边数: {G.number_of_edges()}")
        print(f"  平均度: {actual_avg_degree:.4f} (目标: {self.avg_degree})")
        print(f"  平均度误差: {avg_degree_error:.4f}")
        print(f"  自环数: {nx.number_of_selfloops(G)} (应为0)")
        print(f"  是否连通: {nx.is_connected(G)}")
        
        return G
    
    def generate(self, tolerance: float = 0.08, 
                 degree_dist_tolerance: float = 0.09) -> nx.Graph:
        """
        生成符合要求的无标度网络
        
        参数:
            tolerance: 平均度误差容忍度
            degree_dist_tolerance: 度分布相对误差容忍度
            
        返回:
            生成的网络图
        """
        # 特殊情况：gamma = 3 时使用 BA 模型
        if abs(self.gamma - 3.0) < 1e-6:
            print(f"检测到 γ={self.gamma}，使用 Barabási-Albert (BA) 模型")
            return self._generate_ba_network()
        
        # 其他情况使用配置模型
        best_graph = None
        best_avg_error = float('inf')
        best_dist_error = float('inf')
        best_combined_score = float('inf')
        
        for iteration in range(self.max_iterations):
            # 生成度序列
            degrees = self.generate_degree_sequence()
            
            # 调整度序列以匹配目标平均度
            degrees = self.adjust_degree_sequence(degrees)
            
            # 使用配置模型生成网络
            G = self.configuration_model(degrees)
            
            # 验证生成的网络
            actual_N = G.number_of_nodes()
            actual_degrees = [G.degree(n) for n in G.nodes()]
            actual_avg_degree = np.mean(actual_degrees)
            
            # 检查节点数
            if actual_N != self.N:
                continue
            
            # 验证并清理：确保没有自环和重边
            G = self._ensure_simple_graph(G)
            
            # 重新计算度（因为可能移除了一些边）
            actual_degrees = [G.degree(n) for n in G.nodes()]
            actual_avg_degree = np.mean(actual_degrees)
            
            # 计算平均度误差
            avg_degree_error = abs(actual_avg_degree - self.avg_degree)
            
            # 计算度分布误差
            degree_dist_error = self.check_degree_distribution(actual_degrees)
            
            # 计算综合得分
            normalized_avg_error = avg_degree_error / tolerance
            normalized_dist_error = degree_dist_error / degree_dist_tolerance
            combined_score = normalized_avg_error + normalized_dist_error
            
            # 更新最佳结果（基于综合得分）
            if combined_score < best_combined_score:
                best_combined_score = combined_score
                best_avg_error = avg_degree_error
                best_dist_error = degree_dist_error
                best_graph = G.copy()
            
            # 检查是否同时满足两个条件
            if avg_degree_error <= tolerance and degree_dist_error <= degree_dist_tolerance:
                print(f"✓ 成功生成网络 (迭代次数: {iteration + 1})")
                print(f"  节点数: {actual_N} (严格)")
                print(f"  平均度: {actual_avg_degree:.4f} (目标: {self.avg_degree})")
                print(f"  平均度误差: {avg_degree_error:.4f} (≤ {tolerance})")
                print(f"  度分布相对误差: {degree_dist_error:.4f} (≤ {degree_dist_tolerance})")
                print(f"  综合得分: {combined_score:.4f}")
                print(f"  自环数: {nx.number_of_selfloops(G)} (应为0)")
                
                return G
        
        # 如果达到最大迭代次数，返回综合得分最好的结果
        if best_graph is not None:
            # 最后确保是简单图
            best_graph = self._ensure_simple_graph(best_graph)
            
            actual_degrees = [best_graph.degree(n) for n in best_graph.nodes()]
            actual_avg_degree = np.mean(actual_degrees)
            
            warnings.warn(f"达到最大迭代次数 {self.max_iterations}，返回综合得分最佳结果")
            print(f"! 最佳结果:")
            print(f"  节点数: {best_graph.number_of_nodes()} (严格)")
            print(f"  平均度: {actual_avg_degree:.4f} (目标: {self.avg_degree})")
            print(f"  平均度误差: {best_avg_error:.4f} (目标: ≤ {tolerance})")
            print(f"  度分布相对误差: {best_dist_error:.4f} (目标: ≤ {degree_dist_tolerance})")
            print(f"  综合得分: {best_combined_score:.4f}")
            print(f"  自环数: {nx.number_of_selfloops(best_graph)} (应为0)")
            
            # 给出建议
            if best_avg_error > tolerance:
                print(f"  提示: 平均度误差超标 ({best_avg_error:.4f} > {tolerance})")
            if best_dist_error > degree_dist_tolerance:
                print(f"  提示: 度分布误差超标 ({best_dist_error:.4f} > {degree_dist_tolerance})")
            print(f"  建议: 增加 max_iterations 或放宽容忍度")
            
            return best_graph
        
        raise RuntimeError("无法生成满足条件的网络，请调整参数或增加最大迭代次数")
    
    def check_degree_distribution(self, actual_degrees: list) -> float:
        """
        检查实际度分布与理论分布的相对误差
        
        参数:
            actual_degrees: 实际度序列
            
        返回:
            相对误差
        """
        # 统计度分布
        degree_count = Counter(actual_degrees)
        degrees_unique = sorted(degree_count.keys())
        
        if len(degrees_unique) < 2:
            return float('inf')
        
        # 过滤掉度为0的节点（孤立节点）
        degrees_unique = [k for k in degrees_unique if k > 0]
        
        if len(degrees_unique) < 2:
            return float('inf')
        
        # 计算理论概率
        k_min = max(1, min(degrees_unique))
        k_max = max(degrees_unique)
        
        if k_min <= 0 or k_max <= 0:
            return float('inf')
        
        # 归一化常数（使用浮点数）
        C = sum([float(k)**(-self.gamma) for k in range(k_min, k_max + 1)])
        
        if C <= 0:
            return float('inf')
        
        errors = []
        for k in degrees_unique:
            if k < k_min:
                continue
            
            actual_prob = degree_count[k] / len(actual_degrees)
            theoretical_prob = float(k)**(-self.gamma) / C
            
            if theoretical_prob > 0:
                relative_error = abs(actual_prob - theoretical_prob) / theoretical_prob
                errors.append(relative_error)
        
        return np.mean(errors) if errors else float('inf')
    
    def plot_degree_distribution(self, G: nx.Graph, save_path: Optional[str] = None):
        """
        绘制度分布图
        
        参数:
            G: 网络图
            save_path: 保存路径（可选）
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        degrees = [G.degree(n) for n in G.nodes()]
        degree_count = Counter(degrees)
        
        # 提取度值和频率
        k_values = sorted([k for k in degree_count.keys() if k > 0])
        
        if len(k_values) == 0:
            print("警告: 网络中没有度大于0的节点")
            return
        
        frequencies = [degree_count[k] for k in k_values]
        probabilities = [f / len(degrees) for f in frequencies]
        
        # 理论分布
        k_min = min(k_values)
        k_max = max(k_values)
        k_theory = np.arange(k_min, k_max + 1, dtype=float)  # 使用浮点数
        
        # 计算归一化常数 C（确保使用浮点数运算）
        C = sum([float(k)**(-self.gamma) for k in k_theory])
        p_theory = [float(k)**(-self.gamma) / C for k in k_theory]
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 线性坐标
        ax1.scatter(k_values, probabilities, alpha=0.6, label='实际分布', s=50, color='blue')
        ax1.plot(k_theory, p_theory, 'r-', label=f'理论分布 (γ={self.gamma})', linewidth=2)
        ax1.set_xlabel('度 k', fontsize=12)
        ax1.set_ylabel('概率 P(k)', fontsize=12)
        ax1.set_title('度分布 (线性坐标)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 对数坐标
        ax2.scatter(k_values, probabilities, alpha=0.6, label='实际分布', s=50, color='blue')
        ax2.plot(k_theory, p_theory, 'r-', label=f'理论分布 (γ={self.gamma})', linewidth=2)
        ax2.set_xlabel('度 k', fontsize=12)
        ax2.set_ylabel('概率 P(k)', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title('度分布 (对数坐标)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which="both")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图像已保存到: {save_path}")
        
        plt.show()
    
    def print_network_stats(self, G: nx.Graph):
        """
        打印网络统计信息
        
        参数:
            G: 网络图
        """
        degrees = [G.degree(n) for n in G.nodes()]
        
        print("\n" + "="*50)
        print("网络统计信息")
        print("="*50)
        print(f"节点数: {G.number_of_nodes()}")
        print(f"边数: {G.number_of_edges()}")
        print(f"平均度: {np.mean(degrees):.4f}")
        print(f"度标准差: {np.std(degrees):.4f}")
        print(f"最小度: {min(degrees)}")
        print(f"最大度: {max(degrees)}")
        print(f"孤立节点数: {sum(1 for d in degrees if d == 0)}")
        
        # 检查连通性
        if nx.is_connected(G):
            print(f"网络连通: 是")
        else:
            components = list(nx.connected_components(G))
            print(f"网络连通: 否")
            print(f"连通分量数: {len(components)}")
            print(f"最大连通分量大小: {len(max(components, key=len))}")
        
        print("="*50 + "\n")


if __name__ == "__main__":

    # 设置中文字体，以防乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    generatorSF = ScaleFreeNetworkGenerator(
        N=30000,
        avg_degree=4.0,
        gamma=2.7,
        max_iterations=6000
    )

    print("\n" + "="*50)
    print(f"生成γ={generatorSF.gamma}的网络")
    print("="*50)
    
    # 定义保存路径
    save_dir = "scr/byself/saved_networks"
    network_file = f"{save_dir}/SFA_N{generatorSF.N}_k{generatorSF.avg_degree}_gamma{generatorSF.gamma}.graphml"
    
    # 检查文件是否已存在
    if Path(network_file).exists():
        print(f"\n发现已保存的网络文件，正在加载...")
        GA = NetworkIO.load_graph(network_file, format='graphml')
    else:
        print(f"\n未找到已保存的网络，正在生成...")
        GA = generatorSF.generate()
        
        # 保存网络
        NetworkIO.save_graph(GA, network_file, format='graphml', include_metadata=False)
    
    generatorSF.print_network_stats(GA)
    generatorSF.plot_degree_distribution(GA, save_path='scale_free_network_gamma3.png')