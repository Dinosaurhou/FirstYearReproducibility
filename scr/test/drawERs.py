import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# --- 1. 定义网络参数 ---
N = 200  # 节点数
AVG_DEGREE = 4  # 平均度
P = AVG_DEGREE / (N - 1)  # ER网络中的连接概率

# --- 2. 生成两个ER网络 ---
G_A = nx.erdos_renyi_graph(N, P, seed=42)
G_B = nx.erdos_renyi_graph(N, P, seed=43)

# --- 3. 创建一个组合图 ---
# 为了避免节点标签冲突，我们将网络B的节点重新标记
# 网络A的节点为 0 to 199
# 网络B的节点为 200 to 399
G = nx.Graph()
G.add_nodes_from(G_A.nodes(data=True))
G.add_edges_from(G_A.edges())

mapping_B = {node: node + N for node in G_B.nodes()}
G_B_relabeled = nx.relabel_nodes(G_B, mapping_B)
G.add_nodes_from(G_B_relabeled.nodes(data=True))
G.add_edges_from(G_B_relabeled.edges())


# --- 4. 在两个网络之间添加连接 ---
# 从网络A中随机选择100个节点
nodes_A = list(G_A.nodes())
nodes_to_connect_A = np.random.choice(nodes_A, 100, replace=False)

# 从重新标记的网络B中随机选择100个节点
nodes_B_relabeled = list(G_B_relabeled.nodes())
nodes_to_connect_B = np.random.choice(nodes_B_relabeled, 100, replace=False)

# 添加连接
for i in range(100):
    G.add_edge(nodes_to_connect_A[i], nodes_to_connect_B[i])

# --- 5. 计算节点的三维布局 ---
pos = nx.spring_layout(G, dim=3, seed=42)

# --- 6. 使用Plotly进行3D可视化 ---
# 提取节点位置
node_x = [pos[node][0] for node in G.nodes()]
node_y = [pos[node][1] for node in G.nodes()]
node_z = [pos[node][2] for node in G.nodes()]

# 为不同网络的节点分配颜色
colors = []
for node in G.nodes():
    if node < N:
        colors.append('blue')  # 网络A的节点为蓝色
    else:
        colors.append('red')   # 网络B的节点为红色

# 创建节点轨迹
node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=colors,
        size=5,
        line_width=0.5
    ))

# 创建边的轨迹
edge_x = []
edge_y = []
edge_z = []
for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

# --- 7. 创建图形并显示 ---
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title_text='<br>两个连接的ER网络的三维可视化',
                title_font_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="蓝色: 网络A, 红色: 网络B",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                ))

fig.show()


# 我还在实验一下
# 修改了仓库名字看是否还可以上传