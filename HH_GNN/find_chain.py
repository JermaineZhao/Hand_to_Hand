# GOOOAAAAATTTT


import networkx as nx
import csv

# 从CSV文件中加载节点和边信息
G = nx.DiGraph()

# 加载节点信息
with open('nodes.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        G.add_node(row['node'], type=row['type'])

# 加载边信息
with open('edges.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        similarity = float(row['similarity']) if row['similarity'] else 0.0
        G.add_edge(row['source'], row['target'], relation=row['relation'], similarity=similarity)

# 定义我们要找到的链条中的关键边
required_edges = [
    ('item11', 'user8', 'wants'),
    ('item20', 'user2', 'wants'),
    # ('item1', 'user22', 'wants'),
    # ('user22', 'item49', 'owns'),
    # ('item49', 'user23', 'wants'),
    # ('user23', 'item1', 'owns')
]

# 定义相似度阈值
similarity_threshold = 0.7

def generate_possible_edges(required_edges, G, similarity_threshold):
    possible_edge_sets = [set()]
    for edge in required_edges:
        item, user, relation = edge
        new_edge_sets = []
        for edge_set in possible_edge_sets:
            # 原始边
            new_edge_sets.append(edge_set | {edge})
            # 替代边
            for similar_item in G.successors(item):
                if G[item][similar_item]['relation'] == 'similar' and G[item][similar_item]['similarity'] >= similarity_threshold:
                    new_edge_sets.append(edge_set | {(similar_item, user, relation)})
        possible_edge_sets = new_edge_sets
    return possible_edge_sets

# 优化的DFS函数来找到包含所需边的路径
def find_all_paths_dfs(G, required_edge_sets, similarity_threshold, max_length=6):
    all_paths = []
    
    def dfs(node, start_node, path, remaining_edges, last_type, replaced_nodes):
        # 如果路径长度超过max_length，停止搜索
        if len(path) > max_length:
            return
        
        # 更新路径
        path = path + [node]
        
        # 如果没有剩余的边且可以回到起始节点，保存路径
        if not remaining_edges and G.has_edge(node, start_node):
            all_paths.append((path + [start_node], replaced_nodes))
            return
        
        for neighbor in G.successors(node):
            edge = (node, neighbor, G[node][neighbor]['relation'])
            next_type = 'user' if G.nodes[neighbor]['type'] == 'user' else 'item'
            
            # 确保用户和物品交替出现
            if next_type == last_type:
                continue
            
            if edge in remaining_edges:
                new_remaining_edges = remaining_edges - {edge}
                dfs(neighbor, start_node, path, new_remaining_edges, next_type, replaced_nodes)
            else:
                dfs(neighbor, start_node, path, remaining_edges, next_type, replaced_nodes)
        
        # 检查相似度替代规则
        if G.nodes[node]['type'] == 'item':
            for similar_item, similarity in G[node].items():
                if G.nodes[similar_item]['type'] == 'item' and similarity['relation'] == 'similar' and similarity['similarity'] >= similarity_threshold:
                    for neighbor in G.successors(similar_item):
                        edge = (similar_item, neighbor, G[similar_item][neighbor]['relation'])
                        next_type = 'user' if G.nodes[neighbor]['type'] == 'user' else 'item'
                        
                        # 确保用户和物品交替出现
                        if next_type == last_type:
                            continue
                        
                        if edge in remaining_edges:
                            new_remaining_edges = remaining_edges - {edge}
                            dfs(neighbor, start_node, path, new_remaining_edges, next_type, replaced_nodes | {similar_item})
                        else:
                            dfs(neighbor, start_node, path, remaining_edges, next_type, replaced_nodes | {similar_item})
    
    # 从图中的所有用户节点开始搜索
    start_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'user']
    for required_edges in required_edge_sets:
        for start in start_nodes:
            dfs(start, start, [], required_edges, 'user', set())
    
    return all_paths

# 生成所有可能的边集合
possible_edge_sets = generate_possible_edges(required_edges, G, similarity_threshold)

# 开始搜索
all_paths = find_all_paths_dfs(G, possible_edge_sets, similarity_threshold)

if all_paths:
    for path, replaced_nodes in all_paths:
        formatted_path = " -> ".join([f"{node}(replaced)" if node in replaced_nodes else node for node in path])
        print("Found a path: ", formatted_path)
else:
    print("No path found that includes all required edges.")

