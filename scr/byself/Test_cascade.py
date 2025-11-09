import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import CascadeFailure as cf



G1 = nx.Graph()
G1.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
G1.add_edges_from([(0, 2), (1, 2), (2, 3), (2, 4), (4, 6), (5, 6), (6, 7)])

G2 = nx.Graph()
G2.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
G2.add_edges_from([(0, 2), (1, 2), (1, 4), (1, 5), (2, 3), (5, 6), (5, 7)])

N = len(G1.nodes)

node_mapping = {i: i for i in range(N)}

G1_after, G2_after = cf.cascade_failure_max_specific_nodes(G1, G2, node_mapping, [1])


# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制 G1_after
pos1 = nx.spring_layout(G1_after, seed=42)
nx.draw(G1_after, pos1, ax=ax1, with_labels=True, node_color='lightblue', 
    node_size=500, font_size=12, font_weight='bold', edge_color='gray')
ax1.set_title('G1 after cascade failure')

# 绘制 G2_after
pos2 = nx.spring_layout(G2_after, seed=42)
nx.draw(G2_after, pos2, ax=ax2, with_labels=True, node_color='lightgreen', 
    node_size=500, font_size=12, font_weight='bold', edge_color='gray')
ax2.set_title('G2 after cascade failure')

plt.tight_layout()
plt.show()












