import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import csv

# 创建图
G = nx.DiGraph()

# 添加用户和物品节点
num_users = 30
num_items = 70
users = [f'user{I}' for I in range(1, num_users + 1)]
items = [f'item{I}' for I in range(1, num_items + 1)]

for user in users:
    G.add_node(user, type='user')

for item in items:
    G.add_node(item, type='item')

# 随机添加拥有和需求边
num_owns_edges = 100
num_wants_edges = 100

for _ in range(num_owns_edges):
    user = random.choice(users)
    item = random.choice(items)
    G.add_edge(user, item, relation='owns')

for _ in range(num_wants_edges):
    user = random.choice(users)
    item = random.choice(items)
    G.add_edge(item, user, relation='wants')

# 为所有物品对添加相似度边（即使没有显式相似度边，默认相似度为零）
for i, item1 in enumerate(items):
    for j, item2 in enumerate(items):
        if i != j:
            similarity_value = round(random.uniform(0, 1), 2) if random.random() < 0.5 else 0.0
            G.add_edge(item1, item2, relation='similar', similarity=similarity_value)

# 自定义布局算法，确保相似度高的物品更近，相似度低的物品更远
def custom_layout(G, base_dist=0.1, iter_num=100):
    pos = nx.spring_layout(G, seed=42)
    for _ in range(iter_num):
        for node1, node2 in G.edges():
            if G.edges[node1, node2].get('relation') == 'similar':
                similarity = G.edges[node1, node2]['similarity']
                target_dist = base_dist * (1 - similarity)
                x1, y1 = pos[node1]
                x2, y2 = pos[node2]
                current_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if current_dist < target_dist:
                    angle = np.arctan2(y2 - y1, x2 - x1)
                    pos[node2] = (x1 + target_dist * np.cos(angle), y1 + target_dist * np.sin(angle))
                elif current_dist > target_dist * 2:
                    angle = np.arctan2(y2 - y1, x2 - x1)
                    pos[node2] = (x1 + target_dist * 2 * np.cos(angle), y1 + target_dist * 2 * np.sin(angle))
    return pos

# 获取布局
pos = custom_layout(G)

# 定义节点颜色
node_colors = ['red' if G.nodes[node]['type'] == 'user' else 'blue' for node in G.nodes()]

# 可视化图
plt.figure(figsize=(20, 20))  # 增加图形的尺寸
nx.draw(G, pos, with_labels=False, node_size=300, node_color=node_colors, edge_color="gray")

# 获取边的标签，包括关系和相似度
edge_labels = nx.get_edge_attributes(G, 'relation')
for (u, v, d) in G.edges(data=True):
    if 'similarity' in d:
        edge_labels[(u, v)] += f' ({d["similarity"]})'

# 在图中显示边的标签
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

# plt.show()

# 打印一些生成的数据以进行检查
print("Owns relations:")
for edge in list(G.edges(data=True))[:10]:
    if edge[2]['relation'] == 'owns':
        print(edge)

print("\nWants relations:")
for edge in list(G.edges(data=True))[:10]:
    if edge[2]['relation'] == 'wants':
        print(edge)

print("\nSimilarity relations:")
for edge in list(G.edges(data=True))[:10]:
    if edge[2]['relation'] == 'similar':
        print(edge)

# 将节点信息保存到CSV文件
with open('nodes.csv', 'w', newline='') as csvfile:
    fieldnames = ['node', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for node, data in G.nodes(data=True):
        writer.writerow({'node': node, 'type': data['type']})

# 将边信息保存到CSV文件
with open('edges.csv', 'w', newline='') as csvfile:
    fieldnames = ['source', 'target', 'relation', 'similarity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for u, v, data in G.edges(data=True):
        writer.writerow({'source': u, 'target': v, 
                         'relation': data['relation'], 
                         'similarity': data.get('similarity', '')})

print("节点和边的信息已经保存到 nodes.csv 和 edges.csv 文件中。")