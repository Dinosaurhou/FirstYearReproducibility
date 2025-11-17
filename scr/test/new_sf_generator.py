import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedSFNetworkGenerator:
    """
    改进的无标度网络生成器
    
    核心策略:
    1. 生成连续幂律度序列
    2. 智能调整以匹配目标平均度（保持幂律形状）
    3. 使用配置模型生成网络
    4. 移除自环和重边
    """
    
    def __init__(self, N: int, avg_degree: float, gamma: float, 
                 seed: Optional[int] = None):
        """
        初始化生成器
        
        参数:
            N: 节点数
            avg_degree: 目标平均度
            gamma: 幂律指数
            seed: 随机种子
        """
        self.N = N
        self.avg_degree = avg_degree
        self.gamma = gamma
        
        if seed is not None:
            np.random.seed(seed)
            
    def generate_power_law_degrees(self) -> np.ndarray:
        """
        生成幂律度序列（方法1：使用 networkx 的 powerlaw_sequence）
        
        返回:
            度序列（浮点数）
        """
        # 使用 powerlaw_sequence 生成度序列
        # 注意：这个函数生成的是连续值
        degrees = np.array(nx.utils.powerlaw_sequence(self.N, exponent=self.gamma))
        
        return degrees
    
    def generate_power_law_degrees_custom(self) -> np.ndarray:
        """
        生成幂律度序列（方法2：自定义逆变换采样）
        
        使用逆CDF方法从幂律分布中采样
        更精确地控制幂律指数
        
        返回:
            度序列（浮点数）
        """
        # 计算 k_min（基于理论公式）
        if self.gamma > 2:
            # 对于幂律分布，平均度 <k> ≈ k_min * (gamma-1)/(gamma-2)
            k_min = self.avg_degree * (self.gamma - 2) / (self.gamma - 1)
        else:
            k_min = self.avg_degree * 0.5
        
        k_min = max(1.0, k_min)
        
        # k_max 基于网络规模
        k_max = min(self.N - 1, int(self.avg_degree * 20))
        
        degrees = []
        
        # 逆变换采样
        # 对于截断幂律分布: P(k) ∝ k^(-γ), k ∈ [k_min, k_max]
        # CDF^(-1)(u) = k_min * [(k_max/k_min)^(1-γ) * u + (1-u)]^(1/(1-γ))
        
        alpha = 1 - self.gamma  # 指数转换
        ratio = (k_max / k_min) ** alpha
        
        for _ in range(self.N):
            u = np.random.random()
            k = k_min * (ratio * u + (1 - u)) ** (1 / alpha)
            degrees.append(k)
        
        return np.array(degrees)
    
    def adjust_degrees_to_target_average(self, degrees: np.ndarray) -> np.ndarray:
        """
        调整度序列以匹配目标平均度
        
        策略：使用分层调整，保持幂律分布形状
        
        参数:
            degrees: 原始度序列（浮点数）
            
        返回:
            调整后的度序列（浮点数）
        """
        current_avg = np.mean(degrees)
        
        # 如果当前平均度已经很接近目标，直接返回
        if abs(current_avg - self.avg_degree) < 0.01:
            return degrees
        
        # 方法1：简单缩放（保持分布形状）
        if abs(current_avg - self.avg_degree) / self.avg_degree < 0.2:
            scale_factor = self.avg_degree / current_avg
            degrees_adjusted = degrees * scale_factor
        else:
            # 方法2：分段调整（对偏差较大的情况）
            degrees_adjusted = degrees.copy()
            
            # 迭代调整
            max_iterations = 1000
            for iteration in range(max_iterations):
                current_avg = np.mean(degrees_adjusted)
                diff = self.avg_degree - current_avg
                
                if abs(diff) < 0.01:
                    break
                
                # 根据偏差方向调整
                if diff > 0:  # 需要增加平均度
                    # 优先增加低度节点（它们数量多，调整影响分布小）
                    low_degree_mask = degrees_adjusted < np.median(degrees_adjusted)
                    num_adjust = max(1, int(self.N * 0.1))  # 调整10%的节点
                    
                    indices = np.where(low_degree_mask)[0]
                    if len(indices) > 0:
                        selected = np.random.choice(indices, 
                                                   size=min(num_adjust, len(indices)), 
                                                   replace=False)
                        increment = min(diff / len(selected), 1.0)
                        degrees_adjusted[selected] += increment
                else:  # 需要减少平均度
                    # 优先减少高度节点
                    high_degree_mask = degrees_adjusted > np.median(degrees_adjusted)
                    num_adjust = max(1, int(self.N * 0.1))
                    
                    indices = np.where(high_degree_mask)[0]
                    if len(indices) > 0:
                        selected = np.random.choice(indices, 
                                                   size=min(num_adjust, len(indices)), 
                                                   replace=False)
                        decrement = min(abs(diff) / len(selected), 1.0)
                        degrees_adjusted[selected] -= decrement
        
        return degrees_adjusted
    
    def convert_to_integer_degrees(self, degrees: np.ndarray) -> np.ndarray:
        """
        将浮点度序列转换为整数度序列
        
        策略：使用概率化取整，保持总度数为偶数
        
        参数:
            degrees: 浮点度序列
            
        返回:
            整数度序列
        """
        # 概率化取整：对每个度值，以小数部分为概率向上取整
        degrees_int = np.zeros(self.N, dtype=int)
        
        for i in range(self.N):
            floor_val = int(degrees[i])
            frac_val = degrees[i] - floor_val
            
            # 以 frac_val 的概率向上取整
            if np.random.random() < frac_val:
                degrees_int[i] = floor_val + 1
            else:
                degrees_int[i] = floor_val
        
        # 确保至少为1
        degrees_int = np.maximum(degrees_int, 1)
        
        # 确保总度数为偶数（图论要求）
        total_degree = np.sum(degrees_int)
        if total_degree % 2 != 0:
            # 随机选择一个节点增加1
            idx = np.random.randint(0, self.N)
            degrees_int[idx] += 1
        
        return degrees_int
    
    def fine_tune_integer_degrees(self, degrees_int: np.ndarray) -> np.ndarray:
        """
        精细调整整数度序列，使其更接近目标平均度
        
        参数:
            degrees_int: 整数度序列
            
        返回:
            调整后的整数度序列
        """
        max_iterations = 1000
        
        for iteration in range(max_iterations):
            current_avg = np.mean(degrees_int)
            error = abs(current_avg - self.avg_degree)
            
            # 如果误差已经很小，停止调整
            if error < 0.01:
                break
            
            diff = self.avg_degree - current_avg
            
            if abs(diff) < 0.01:
                break
            
            if diff > 0:  # 需要增加平均度
                # 选择度较低的节点增加度
                low_degree_indices = np.where(degrees_int < np.median(degrees_int))[0]
                if len(low_degree_indices) > 0:
                    num_to_adjust = max(1, int(abs(diff) * self.N))
                    selected = np.random.choice(low_degree_indices, 
                                              size=min(num_to_adjust, len(low_degree_indices)),
                                              replace=False)
                    degrees_int[selected] += 1
            else:  # 需要减少平均度
                # 选择度较高的节点减少度（但保持至少为1）
                high_degree_indices = np.where((degrees_int > np.median(degrees_int)) & 
                                              (degrees_int > 1))[0]
                if len(high_degree_indices) > 0:
                    num_to_adjust = max(1, int(abs(diff) * self.N))
                    selected = np.random.choice(high_degree_indices,
                                              size=min(num_to_adjust, len(high_degree_indices)),
                                              replace=False)
                    degrees_int[selected] -= 1
            
            # 确保总度数为偶数
            total_degree = np.sum(degrees_int)
            if total_degree % 2 != 0:
                idx = np.random.randint(0, self.N)
                if diff > 0:
                    degrees_int[idx] += 1
                elif degrees_int[idx] > 1:
                    degrees_int[idx] -= 1
                else:
                    degrees_int[idx] += 1
        
        return degrees_int
    
    def configuration_model_simple_graph(self, degree_sequence: np.ndarray, 
                                        max_tries: int = 1000) -> nx.Graph:
        """
        使用配置模型生成简单图（无自环、无重边）
        
        参数:
            degree_sequence: 度序列
            max_tries: 最大尝试次数
            
        返回:
            生成的简单图
        """
        best_G = None
        best_num_edges = 0
        
        for attempt in range(max_tries):
            # 使用配置模型生成
            G = nx.configuration_model(degree_sequence, create_using=nx.Graph())
            
            # 转换为简单图（自动移除自环和重边）
            G = nx.Graph(G)
            
            # 手动移除自环（双重保险）
            G.remove_edges_from(nx.selfloop_edges(G))
            
            # 记录边数最多的图
            if G.number_of_edges() > best_num_edges:
                best_num_edges = G.number_of_edges()
                best_G = G.copy()
            
            # 如果节点数和度序列基本匹配，接受这个图
            if G.number_of_nodes() == self.N:
                actual_avg_degree = 2 * G.number_of_edges() / self.N
                if abs(actual_avg_degree - self.avg_degree) / self.avg_degree < 0.15:
                    break
        
        return best_G
    
    def verify_power_law(self, G: nx.Graph) -> Tuple[float, float]:
        """
        验证网络的幂律特性
        
        使用对数线性回归估计幂律指数
        
        参数:
            G: 网络图
            
        返回:
            (estimated_gamma, relative_error)
        """
        degrees = [d for n, d in G.degree()]
        
        # 统计度分布
        degree_counts = np.bincount(degrees)
        k_values = np.arange(len(degree_counts))
        pk = degree_counts / np.sum(degree_counts)
        
        # 过滤有效数据点
        mask = (pk > 0) & (k_values >= 2)
        k_values = k_values[mask]
        pk = pk[mask]
        
        if len(k_values) < 5:
            return 0.0, float('inf')
        
        # 对数线性回归
        log_k = np.log(k_values)
        log_pk = np.log(pk)
        
        # 检查有效性
        if np.any(~np.isfinite(log_k)) or np.any(~np.isfinite(log_pk)):
            return 0.0, float('inf')
        
        try:
            # 使用最小二乘法拟合
            coeffs = np.polyfit(log_k, log_pk, 1)
            estimated_gamma = -coeffs[0]
            
            relative_error = abs(estimated_gamma - self.gamma) / self.gamma
            
            return estimated_gamma, relative_error
        except:
            return 0.0, float('inf')
    
    def generate(self, method: str = 'custom', max_attempts: int = 50) -> nx.Graph:
        """
        生成SF网络
        
        参数:
            method: 度序列生成方法 ('networkx' 或 'custom')
            max_attempts: 最大尝试次数
            
        返回:
            生成的网络
        """
        print(f"开始生成SF网络: N={self.N}, <k>={self.avg_degree}, γ={self.gamma}")
        print(f"度序列生成方法: {method}")
        print("="*60)
        
        best_G = None
        best_avg_error = float('inf')
        best_gamma_error = float('inf')
        best_score = float('inf')
        
        for attempt in range(max_attempts):
            # 步骤1: 生成幂律度序列（浮点数）
            if method == 'networkx':
                degrees_float = self.generate_power_law_degrees()
            else:  # 'custom'
                degrees_float = self.generate_power_law_degrees_custom()
            
            # 步骤2: 调整度序列以匹配目标平均度
            degrees_adjusted = self.adjust_degrees_to_target_average(degrees_float)
            
            # 步骤3: 转换为整数度序列
            degrees_int = self.convert_to_integer_degrees(degrees_adjusted)
            
            # 步骤4: 精细调整整数度序列
            degrees_final = self.fine_tune_integer_degrees(degrees_int)
            
            # 步骤5: 使用配置模型生成网络
            G = self.configuration_model_simple_graph(degrees_final)
            
            if G is None or G.number_of_nodes() != self.N:
                continue
            
            # 计算误差
            actual_avg_degree = 2 * G.number_of_edges() / self.N
            avg_error = abs(actual_avg_degree - self.avg_degree)
            
            estimated_gamma, gamma_error = self.verify_power_law(G)
            
            # 综合得分
            score = avg_error / 0.1 + gamma_error / 0.1
            
            # 更新最佳结果
            if score < best_score:
                best_score = score
                best_avg_error = avg_error
                best_gamma_error = gamma_error
                best_G = G.copy()
            
            # 检查是否满足要求
            if avg_error <= 0.1 and gamma_error <= 0.1:
                print(f"✓ 生成成功 (第{attempt+1}次尝试)")
                print(f"  实际平均度: {actual_avg_degree:.4f}")
                print(f"  平均度误差: {avg_error:.4f}")
                print(f"  估计γ: {estimated_gamma:.4f}")
                print(f"  γ相对误差: {gamma_error:.4f}")
                print(f"  自环数: {nx.number_of_selfloops(G)}")
                return G
        
        # 返回最佳结果
        if best_G is not None:
            actual_avg_degree = 2 * best_G.number_of_edges() / self.N
            estimated_gamma, _ = self.verify_power_law(best_G)
            
            print(f"! 达到最大尝试次数，返回最佳结果:")
            print(f"  实际平均度: {actual_avg_degree:.4f}")
            print(f"  平均度误差: {best_avg_error:.4f} (目标: ≤0.1)")
            print(f"  估计γ: {estimated_gamma:.4f}")
            print(f"  γ相对误差: {best_gamma_error:.4f} (目标: ≤0.1)")
            print(f"  自环数: {nx.number_of_selfloops(best_G)}")
            
            return best_G
        
        raise RuntimeError("无法生成满足条件的网络")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例1: 生成 N=1000, <k>=6, γ=2.5 的网络
    print("示例1: 生成小规模网络")
    print("="*60)
    
    generator1 = ImprovedSFNetworkGenerator(N=30000, avg_degree=4, gamma=2.7)
    G1 = generator1.generate(method='custom', max_attempts=3000)

    # 保存网络为graphml格式
    nx.write_graphml(G1, "SFB_27.graphml")
    print(f"\n✓ 网络已保存为 SFB_27.graphml")

    print(f"\n最终网络统计:")
    print(f"  节点数: {G1.number_of_nodes()}")
    print(f"  边数: {G1.number_of_edges()}")
    print(f"  平均度: {2*G1.number_of_edges()/G1.number_of_nodes():.4f}")
    print(f"  是否连通: {nx.is_connected(G1)}")
    
    # 绘制度分布
    degrees = [d for n, d in G1.degree()]
    degree_counts = np.bincount(degrees)
    k_values = np.arange(len(degree_counts))
    pk = degree_counts / np.sum(degree_counts)
    
    mask = (pk > 0) & (k_values > 0)
    k_plot = k_values[mask]
    pk_plot = pk[mask]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(k_plot, pk_plot, 'o', markersize=8, alpha=0.7, label='实际数据')
    
    # 理论曲线
    k_theory = np.logspace(np.log10(k_plot.min()), np.log10(k_plot.max()), 100)
    C_fit = np.exp(np.mean(np.log(pk_plot) + 2.5 * np.log(k_plot)))
    pk_theory = C_fit * (k_theory ** -2.5)
    plt.loglog(k_theory, pk_theory, '--', linewidth=2, label='理论 $k^{-2.7}$', color='red')
    
    plt.xlabel('度 k', fontsize=12, fontweight='bold')
    plt.ylabel('概率 P(k)', fontsize=12, fontweight='bold')
    plt.title('SF网络度分布验证', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('sf_network_verification.png', dpi=300)
    print("\n✓ 度分布图已保存为 sf_network_verification.png")
    plt.show()