import networkx as nx
import matplotlib.pyplot as plt

# 生成ER网络
# N = 100 节点数
# 平均度 = 4，对于ER网络，p = <k>/(N-1)
N = 100
average_degree = 4
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


# 可视化两个网络
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 网络A
pos_A = nx.spring_layout(G_A, seed=42)
nx.draw(G_A, pos_A, ax=axes[0], node_size=150, node_color='lightblue', 
        edge_color='gray', with_labels=True, alpha=0.7)
axes[0].set_title(f"网络A - ER网络 (N={N}, 平均度≈{average_degree})")

# 网络B
pos_B = nx.spring_layout(G_B, seed=42)
nx.draw(G_B, pos_B, ax=axes[1], node_size=150, node_color='lightcoral', 
        edge_color='gray', with_labels=True, alpha=0.7)
axes[1].set_title(f"网络B - ER网络 (N={N}, 平均度≈{average_degree})")

plt.tight_layout()
# plt.savefig('fig.png', bbox_inches='tight') # 替换 plt.show()

plt.show()