# GOOOOOOAAAATTTTT
import pandas as pd
import numpy as np
import random
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F

nodes = pd.read_csv('nodes_1.csv')
edges = pd.read_csv('edges_1.csv')

# 为物品描述信息生成TF-IDF特征
descriptions = nodes[nodes['description'].notna()]['description']
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions).toarray()

# 创建一个字典来存储物品的TF-IDF特征
tfidf_dict = dict(zip(nodes[nodes['description'].notna()]['node'], tfidf_matrix))

# 确定TF-IDF特征的维度
tfidf_dim = tfidf_matrix.shape[1]

# 创建图
G = nx.DiGraph()

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
    G.add_edge(row['source'], row['target'], relation=row['relation'])

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
# Initialize some edge labels to 1 for diversity
for i in range(0, len(edge_labels), 10):
    edge_labels[i] = 1


for u, v, data in G.edges(data=True):
    edges_list.append((list(G.nodes()).index(u), list(G.nodes()).index(v)))
    edge_labels.append(0)  # 这里我们没有相似性标签，所有边标签初始化为0

edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_labels, dtype=torch.float).view(-1, 1)

# 创建图数据
data = Data(x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr)

print("Node features shape:", data.x.shape)
print("Edge index shape:", data.edge_index.shape)
print("Edge attributes shape:", data.edge_attr.shape)


# 定义GCN模型来预测边的特征
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.conv3 = GCNConv(hidden_channels2, out_channels)
        self.fc = torch.nn.Linear(2 * out_channels, 1)  # 新增全连接层，将拼接后的边特征映射到单个输出

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

    def edge_features(self, edge_index, x):
        # 拼接边的两个节点的特征
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        return self.fc(edge_features)  # 使用全连接层将拼接后的特征映射到单个输出
    
# 初始化模型、损失函数和优化器
model = GCN(data.x.shape[1], 32, 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


# Initialize some edge labels to 1 for diversity
for i in range(0, len(edge_labels), 10):
    edge_labels[i] = 1

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    edge_features = model.edge_features(data.edge_index, out).view(-1, 1)
    loss = criterion(edge_features, data.edge_attr)
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the model with more epochs
num_epochs = 5  # 增加训练轮次
for epoch in range(num_epochs):
    loss = train(data)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')


# Predict item similarities
model.eval()

# Create item pair combinations (considering only item-item edges)
item_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'item']
item_combinations = [(i, j) for i in item_nodes for j in item_nodes if i != j]

# Predict similarities for item pairs
predicted_similarities = {}
for item1, item2 in item_combinations:
    node_indices = [list(G.nodes()).index(item1), list(G.nodes()).index(item2)]
    edge_index_test = torch.tensor([node_indices, node_indices[::-1]], dtype=torch.long).t().contiguous()
    data_test = Data(x=data.x, edge_index=edge_index_test)
    with torch.no_grad():
        out = model(data_test.x, data_test.edge_index)
        similarity = torch.sigmoid(model.edge_features(data_test.edge_index, out)).view(-1)
        predicted_similarities[(item1, item2)] = similarity.mean().item() * 10000  # Multiply similarity by 10000


# Output predicted similarities for the current epoch
# Output predicted similarities for the current epoch
print(f"Epoch {epoch} similarities (multiplied by 1000):")
for (item1, item2), similarity in predicted_similarities.items():
    if similarity > 300:  # Only print similarities above the threshold
        description1 = nodes[nodes['node'] == item1]['description'].values[0]
        description2 = nodes[nodes['node'] == item2]['description'].values[0]
        print(f"Predicted similarity between {item1} and {item2}: {similarity:.4f}")
        print(f" - {item1} description: {description1}")
        print(f" - {item2} description: {description2}")