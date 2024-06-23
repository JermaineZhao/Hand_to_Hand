import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F

# 生成示例数据
nodes = pd.DataFrame({
    'node': ['user1', 'user2', 'item1', 'item2', 'item3', 'item4'],
    'type': ['user', 'user', 'item', 'item', 'item', 'item'],
    'age': [25, 30, None, None, None, None],
    'gender': [1, 0, None, None, None, None],  # 性别：1表示男，0表示女
    'category': [None, None, 'electronics', 'books', 'electronics', 'clothing'],
    'price': [None, None, 299.99, 19.99, 399.99, 49.99],
    'description': [None, None, 'Smartphone with 6GB RAM', 'Fiction book', '4K Television', 'T-shirt']
})

edges = pd.DataFrame({
    'source': ['user1', 'user1', 'user2', 'user2', 'item1', 'item2'],
    'target': ['item1', 'item2', 'item2', 'item3', 'item2', 'item4'],
    'relation': ['buys', 'views', 'buys', 'rates', 'similar', 'similar'],
    'similarity': [None, None, None, None, None, None]  # 仅相似关系有相似度
})

# 创建图
G = nx.DiGraph()

# 为物品描述信息生成TF-IDF特征
descriptions = nodes[nodes['description'].notna()]['description']
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions).toarray()

# 创建一个字典来存储物品的TF-IDF特征
tfidf_dict = dict(zip(nodes[nodes['description'].notna()]['node'], tfidf_matrix))

# 确定TF-IDF特征的维度
tfidf_dim = tfidf_matrix.shape[1]

# 添加节点并设置特征
for _, row in nodes.iterrows():
    if row['type'] == 'user':
        feature = [row['age'] if row['age'] is not None else 0, row['gender'] if row['gender'] is not None else 0]
    else:
        category_feature = [
            1 if row['category'] == 'electronics' else 0,
            1 if row['category'] == 'books' else 0,
            1 if row['category'] == 'clothing' else 0
        ]
        price_feature = [row['price']] if row['price'] is not None else [0]
        description_feature = tfidf_dict[row['node']] if row['node'] in tfidf_dict else np.zeros(tfidf_dim)
        feature = category_feature + price_feature + list(description_feature)
    
    G.add_node(row['node'], type=row['type'], feature=feature)

# 添加边
for _, row in edges.iterrows():
    similarity = float(row['similarity']) if row['similarity'] is not None else 0.0
    G.add_edge(row['source'], row['target'], relation=row['relation'], similarity=similarity)

# 确保所有节点的特征维度一致
max_feature_length = max(len(G.nodes[node]['feature']) for node in G.nodes())
for node in G.nodes():
    feature = G.nodes[node]['feature']
    if len(feature) < max_feature_length:
        G.nodes[node]['feature'] = feature + [0] * (max_feature_length - len(feature))

# 提取节点特征和标签
node_features = np.array([G.nodes[node]['feature'] for node in G.nodes()])
node_labels = np.array([1 if G.nodes[node]['type'] == 'user' else 0 for node in G.nodes()], dtype=int)

# 提取边特征和边标签
edges_list = []
edge_labels = []
for u, v, data in G.edges(data=True):
    edges_list.append((list(G.nodes()).index(u), list(G.nodes()).index(v)))
    edge_labels.append(data['relation'] == 'similar')

edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_labels, dtype=torch.float).view(-1, 1)

# 创建图数据
data = Data(x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr)

print("Node features shape:", data.x.shape)
print("Edge index shape:", data.edge_index.shape)
print("Edge attributes shape:", data.edge_attr.shape)

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 初始化模型、损失函数和优化器
model = GCN(data.x.shape[1], 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# 训练函数
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data).view(-1, 1)
    loss = criterion(out, data.edge_attr)
    loss.backward()
    optimizer.step()
    return loss.item()

# 训练模型
for epoch in range(200):
    loss = train(data)
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')


model.eval()
with torch.no_grad():
    out = model(data)
    predictions = torch.sigmoid(out).squeeze().numpy()

# 根据预测结果和相似性阈值筛选相似边
similarity_threshold = 0.7
predicted_similar_edges = [(list(G.nodes())[u], list(G.nodes())[v]) for i, (u, v) in enumerate(edges_list) if predictions[i] >= similarity_threshold]

print("Predicted similar edges:", predicted_similar_edges)