import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 读取CSV文件
edges = pd.read_csv('edges.csv')
nodes = pd.read_csv('nodes.csv')

# 创建空的NetworkX图
G = nx.Graph()

# 提取item nodes的相似性
item_similarity = edges[edges['relation'] == 'similar']

# 正规化similarity值并增加对比度
if not item_similarity.empty:
    similarity_values = item_similarity['similarity'].values
    # 增加对比度
    stretched_similarity = np.interp(similarity_values, (similarity_values.min(), similarity_values.max()), (0, 200))
    norm = plt.Normalize(vmin=stretched_similarity.min(), vmax=stretched_similarity.max())
    item_colors = {node: cm.inferno(norm(stretched_similarity[i])) for i, node in enumerate(item_similarity['source'])}
else:
    item_colors = {}

# 添加节点，区分item nodes和user nodes
for _, row in nodes.iterrows():
    if row['type'] == 'item':
        color = item_colors.get(row['node'], 'blue')
        G.add_node(row['node'], color=color, size=300)
    elif row['type'] == 'user':
        G.add_node(row['node'], color='red', size=300)

# 添加边，标注wants和owns
for _, row in edges.iterrows():
    if row['relation'] in ['wants', 'owns']:
        G.add_edge(row['source'], row['target'], relationship=row['relation'])

# 布局算法：item nodes之间的距离按照similarity来表示
pos = nx.spring_layout(G, k=0.15, iterations=20)

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(24, 12))

# 定义路径
path = ['user7', 'item11', 'user8', 'item19', 'user2', 'item13', 'user7']
path_edges = list(zip(path, path[1:]))

# 绘制不标注路径的图
nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[n]['color'] for n in G.nodes], node_size=[G.nodes[n]['size'] for n in G.nodes], ax=axes[0])
nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['relationship'] in ['wants', 'owns']], ax=axes[0])
nx.draw_networkx_labels(G, pos, font_size=12, ax=axes[0])
axes[0].set_title('Graph Without Path')

# 绘制标注路径的图
nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[n]['color'] for n in G.nodes], node_size=[G.nodes[n]['size'] for n in G.nodes], ax=axes[1])
nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['relationship'] in ['wants', 'owns']], ax=axes[1])
nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=10, edge_color='orange', ax=axes[1])
nx.draw_networkx_labels(G, pos, font_size=12, ax=axes[1])
axes[1].set_title('Graph With Path')

# 添加图例
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Item Nodes')
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='User Nodes')
fig.legend(handles=[blue_patch, red_patch], loc='upper right')

plt.show()