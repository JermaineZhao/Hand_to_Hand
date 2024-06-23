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
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(similarity_df.values)

# 或者使用PCA进行降维
# pca = PCA(n_components=2)
# tsne_results = pca.fit_transform(similarity_df.values)

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
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=False, node_size=50, edge_color='gray', alpha=0.6)
plt.title("Product Similarity Visualization")
plt.show()