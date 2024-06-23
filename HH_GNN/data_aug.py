import networkx as nx
import random
import csv

# 创建图
G = nx.DiGraph()

# 添加用户和物品节点
num_users = 10
num_items = 20
num_owns_edges = 20
num_wants_edges = 20

users = [f'user{I}' for I in range(1, num_users + 1)]
items = [f'item{I}' for I in range(1, num_items + 1)]

for user in users:
    G.add_node(user, type='user')

for item in items:
    G.add_node(item, type='item')

# 使用集合来跟踪每个物品是否已经被拥有
owned_items = set()

# 随机添加拥有和需求边，确保每个item只能被一个user拥有
own_edges = 0
while own_edges < num_owns_edges:
    print(own_edges)
    user = random.choice(users)
    item = random.choice(items)
    if item not in owned_items:
        owned_items.add(item)
        G.add_edge(user, item, relation='owns')
        own_edges += 1

# 添加需求边
want_edges = 0
print(1)
while want_edges < num_wants_edges:
    user = random.choice(users)
    item = random.choice(items)
    G.add_edge(item, user, relation='wants')
    want_edges += 1

print(2)
# 为所有物品对添加相似度边（即使没有显式相似度边，默认相似度为零）
for i, item1 in enumerate(items):
    for j, item2 in enumerate(items):
        if i != j:
            similarity_value = round(random.uniform(0, 1), 2) if random.random() < 0.5 else 0.0
            G.add_edge(item1, item2, relation='similar', similarity=similarity_value)

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