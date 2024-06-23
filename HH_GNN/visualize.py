# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # 读取CSV文件
# edges = pd.read_csv('edges.csv')
# nodes = pd.read_csv('nodes.csv')

# # 创建空的NetworkX图
# G = nx.Graph()

# # 添加节点，区分item nodes和user nodes
# for _, row in nodes.iterrows():
#     if row['type'] == 'item':
#         G.add_node(row['node'], color='blue', size=300)
#     elif row['type'] == 'user':
#         G.add_node(row['node'], color='red', size=300)

# # 添加边，标注wants和owns
# for _, row in edges.iterrows():
#     if row['relation'] in ['wants', 'owns']:
#         G.add_edge(row['source'], row['target'], relationship=row['relation'])

# # 布局算法：item nodes之间的距离按照similarity来表示
# pos = nx.spring_layout(G, k=0.15, iterations=20)

# # 计算item nodes之间的距离
# for _, row in edges.iterrows():
#     if row['relation'] == 'similarity':
#         if row['source'] in pos and row['target'] in pos:
#             dx = pos[row['source']][0] - pos[row['target']][0]
#             dy = pos[row['source']][1] - pos[row['target']][1]
#             distance = (dx**2 + dy**2)**0.5
#             similarity = row['similarity']
#             desired_distance = 1 / similarity if similarity > 0 else 100
#             scale = desired_distance / distance
#             pos[row['target']] = (pos[row['source']][0] + dx * scale, pos[row['source']][1] + dy * scale)

# # 画图
# plt.figure(figsize=(12, 12))
# colors = nx.get_node_attributes(G, 'color').values()
# sizes = nx.get_node_attributes(G, 'size').values()

# # 画节点
# nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes)

# # 画边并标注wants和owns
# labels = nx.get_edge_attributes(G, 'relationship')
# nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in G.edges(data=True) if d['relationship'] in ['wants', 'owns']])
# nx.draw_networkx_labels(G, pos, font_size=12)

# # 添加图例
# blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Item Nodes')
# red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='User Nodes')
# plt.legend(handles=[blue_patch, red_patch])

# # 标注路径
# path = ['user7', 'item11', 'user8', 'item19', 'user2', 'item13', 'user7']
# path_edges = list(zip(path, path[1:]))
# # nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=10, edge_color='orange')

# # 添加节点标签
# nx.draw_networkx_labels(G, pos, font_size=12)

# plt.title('Graph Visualization of Items and Users')
# plt.show()

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取CSV文件
edges = pd.read_csv('edges.csv')
nodes = pd.read_csv('nodes.csv')

# 创建空的NetworkX图
G = nx.Graph()

# 添加节点，区分item nodes和user nodes
for _, row in nodes.iterrows():
    if row['type'] == 'item':
        G.add_node(row['node'], color='blue', size=300)
    elif row['type'] == 'user':
        G.add_node(row['node'], color='red', size=300)

# 添加边，标注wants和owns
for _, row in edges.iterrows():
    if row['relation'] in ['wants', 'owns']:
        G.add_edge(row['source'], row['target'], relationship=row['relation'])

# 布局算法：item nodes之间的距离按照similarity来表示
pos = nx.spring_layout(G, k=0.15, iterations=20)

# 计算item nodes之间的距离
for _, row in edges.iterrows():
    if row['relation'] == 'similarity':
        if row['source'] in pos and row['target'] in pos:
            dx = pos[row['source']][0] - pos[row['target']][0]
            dy = pos[row['source']][1] - pos[row['target']][1]
            distance = (dx**2 + dy**2)**0.5
            similarity = row['similarity']
            desired_distance = 1 / similarity if similarity > 0 else 100
            scale = desired_distance / distance
            pos[row['target']] = (pos[row['source']][0] + dx * scale, pos[row['source']][1] + dy * scale)

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