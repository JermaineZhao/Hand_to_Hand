import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 1. 加载数据
similarity_df = pd.read_csv('product_similarity_2.csv', index_col=0)

# 2. 降维
# 使用t-SNE进行降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
tsne_results = tsne.fit_transform(similarity_df.values)

# 3. 创建图结构
G = nx.Graph()

# 添加节点
num_items = similarity_df.shape[0]
for i in range(num_items):
    G.add_node(i, pos=(tsne_results[i, 0], tsne_results[i, 1]))

# 添加边（只添加相似度较高的边）
threshold = 0.9  # 设定一个相似度阈值，只添加高于该阈值的边
for i in range(num_items):
    for j in range(i+1, num_items):
        if similarity_df.iloc[i, j] > threshold:
            G.add_edge(i, j, weight=similarity_df.iloc[i, j])

# 4. 可视化图形
pos = nx.get_node_attributes(G, 'pos')
edges = G.edges(data=True)

# 绘制节点
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.6)

# 绘制边，边的颜色和宽度表示相似度
for edge in edges:
    node1, node2, weight = edge
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(node1, node2)],
        width=weight['weight']*2,  # 边宽度与相似度成比例
        alpha=0.5,
        edge_color='gray'
    )

plt.title("Product Similarity Visualization")
plt.show()