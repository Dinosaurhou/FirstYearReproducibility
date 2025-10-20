import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

def get_giant_component(G):
    """返回图G的最大连通分量"""
    if not G.nodes():
        return nx.Graph()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = G.subgraph(components[0])
    return giant

def run_simulation_and_get_history(G_A, G_B, initial_attack_fraction):
    """
    运行完整的级联失效模拟，并返回所有步骤的状态历史。
    """
    history = []
    net_A, net_B = G_A.copy(), G_B.copy()
    N = len(G_A.nodes())

    # --- 1. 初始状态 ---
    history.append({
        "net_A": net_A.copy(), "net_B": net_B.copy(),
        "title": "Step 0: 初始网络", "failed_A": set(), "failed_B": set()
    })

    # --- 2. 初始攻击 ---
    nodes_to_remove_A = set(np.random.choice(net_A.nodes(), size=int(N * initial_attack_fraction), replace=False))
    net_A.remove_nodes_from(nodes_to_remove_A)
    history.append({
        "net_A": net_A.copy(), "net_B": net_B.copy(),
        "title": f"Step 1: 初始攻击 {len(nodes_to_remove_A)} 个节点",
        "failed_A": nodes_to_remove_A, "failed_B": set()
    })
    
    last_A_count, last_B_count = len(net_A.nodes()), len(net_B.nodes())
    step = 2

    # --- 3. 级联失效循环 ---
    while True:
        # a. A网络内部失效
        giant_A = get_giant_component(net_A)
        internal_failed_A = set(net_A.nodes()) - set(giant_A.nodes())
        if internal_failed_A:
            net_A.remove_nodes_from(internal_failed_A)
            history.append({"net_A": net_A.copy(), "net_B": net_B.copy(), "title": f"Step {step}: A网络内部失效", "failed_A": internal_failed_A, "failed_B": set()})
            step += 1

        # b. A -> B 跨网络失效
        failed_in_A = set(G_A.nodes()) - set(net_A.nodes())
        dependent_failed_B = set(net_B.nodes()) & failed_in_A
        if dependent_failed_B:
            net_B.remove_nodes_from(dependent_failed_B)
            history.append({"net_A": net_A.copy(), "net_B": net_B.copy(), "title": f"Step {step}: B因A失效而失效", "failed_A": set(), "failed_B": dependent_failed_B})
            step += 1

        # c. B网络内部失效
        giant_B = get_giant_component(net_B)
        internal_failed_B = set(net_B.nodes()) - set(giant_B.nodes())
        if internal_failed_B:
            net_B.remove_nodes_from(internal_failed_B)
            history.append({"net_A": net_A.copy(), "net_B": net_B.copy(), "title": f"Step {step}: B网络内部失效", "failed_A": set(), "failed_B": internal_failed_B})
            step += 1

        # d. B -> A 跨网络失效
        failed_in_B = set(G_B.nodes()) - set(net_B.nodes())
        dependent_failed_A = set(net_A.nodes()) & failed_in_B
        if dependent_failed_A:
            net_A.remove_nodes_from(dependent_failed_A)
            history.append({"net_A": net_A.copy(), "net_B": net_B.copy(), "title": f"Step {step}: A因B失效而失效", "failed_A": dependent_failed_A, "failed_B": set()})
            step += 1
            
        # 检查是否稳定
        if len(net_A.nodes()) == last_A_count and len(net_B.nodes()) == last_B_count:
            history.append({"net_A": net_A.copy(), "net_B": net_B.copy(), "title": "Final Step: 网络达到稳定状态", "failed_A": set(), "failed_B": set()})
            break
        
        last_A_count, last_B_count = len(net_A.nodes()), len(net_B.nodes())
    
    return history

class CascadeVisualizer:
    def __init__(self, G_A, G_B, history):
        self.G_A, self.G_B = G_A, G_B
        self.history = history
        self.current_step = 0

        # 准备固定的3D布局
        self.pos_A = nx.spring_layout(G_A, dim=3, seed=42)
        self.pos_B = {node: np.array([x + 2, y, z]) for node, (x, y, z) in nx.spring_layout(G_B, dim=3, seed=43).items()}
        
        # 创建图形和坐标轴
        self.fig = plt.figure(figsize=(16, 9))
        # 主绘图区
        self.ax = self.fig.add_axes([0, 0.1, 1, 0.9], projection='3d')
        # 按钮区域
        ax_prev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        ax_next = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])

        # 创建按钮
        self.btn_prev = Button(ax_prev, '上一步 (Previous)')
        self.btn_next = Button(ax_next, '下一步 (Next)')
        self.btn_prev.on_clicked(self.prev_step)
        self.btn_next.on_clicked(self.next_step)

        self.draw_step()

    def draw_step(self):
        self.ax.cla()
        state = self.history[self.current_step]
        net_A, net_B = state["net_A"], state["net_B"]
        failed_A, failed_B = state["failed_A"], state["failed_B"]

        # 绘制网络A
        pos_A_alive = {k: v for k, v in self.pos_A.items() if k in net_A}
        if pos_A_alive:
            node_xyz = np.array(list(pos_A_alive.values()))
            self.ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=50, c='skyblue')
        for edge in net_A.edges():
            p1, p2 = self.pos_A[edge[0]], self.pos_A[edge[1]]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='c', alpha=0.5)

        # 绘制网络B
        pos_B_alive = {k: v for k, v in self.pos_B.items() if k in net_B}
        if pos_B_alive:
            node_xyz = np.array(list(pos_B_alive.values()))
            self.ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=50, c='lightgreen')
        for edge in net_B.edges():
            p1, p2 = self.pos_B[edge[0]], self.pos_B[edge[1]]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='g', alpha=0.5)

        # 标记失效节点
        pos_A_failed = {k: v for k, v in self.pos_A.items() if k in failed_A}
        if pos_A_failed:
            node_xyz = np.array(list(pos_A_failed.values()))
            self.ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=80, c='red', marker='x')
        pos_B_failed = {k: v for k, v in self.pos_B.items() if k in failed_B}
        if pos_B_failed:
            node_xyz = np.array(list(pos_B_failed.values()))
            self.ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=80, c='red', marker='x')

        # 绘制依赖关系
        for node in set(net_A.nodes()) & set(net_B.nodes()):
            p_A, p_B = self.pos_A[node], self.pos_B[node]
            self.ax.plot([p_A[0], p_B[0]], [p_A[1], p_B[1]], [p_A[2], p_B[2]], 'r--', alpha=0.3)

        self.ax.set_title(state["title"], fontsize=12)
        self.ax.set_xticks([]); self.ax.set_yticks([]); self.ax.set_zticks([])
        self.fig.suptitle("3D级联失效过程可视化 (红色'x'为本步失效节点)", fontsize=16)
        plt.draw()

    def next_step(self, event):
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            self.draw_step()

    def prev_step(self, event):
        if self.current_step > 0:
            self.current_step -= 1
            self.draw_step()

if __name__ == '__main__':
    N, AVG_DEGREE = 50, 4
    P = AVG_DEGREE / (N - 1)
    seed_A, seed_B = 42, 43
    
    while True:
        G_A = nx.erdos_renyi_graph(N, P, seed=seed_A)
        G_B = nx.erdos_renyi_graph(N, P, seed=seed_B)
        if nx.is_connected(G_A) and nx.is_connected(G_B):
            print(f"成功生成两个连通网络 (seeds: {seed_A}, {seed_B})。")
            break
        seed_A += 1; seed_B += 1
        print(f"网络不连通，正在尝试新的seeds: {seed_A}, {seed_B}...")

    # 1. 运行模拟并获取历史记录
    history = run_simulation_and_get_history(G_A, G_B, initial_attack_fraction=0.1)
    
    # 2. 创建可视化器实例
    visualizer = CascadeVisualizer(G_A, G_B, history)
    
    # 3. 显示窗口
    plt.show()