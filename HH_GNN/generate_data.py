import pandas as pd
import numpy as np
import random

# 生成示例数据
num_users = 30
num_items = 70
users = [f'user{idx + 1}' for idx in range(num_users)]
items = [f'item{idx + 1}' for idx in range(num_items)]

user_data = {
    'node': users,
    'type': ['user'] * num_users,
    'age': np.random.randint(18, 60, size=num_users).tolist(),
    'gender': np.random.randint(0, 2, size=num_users).tolist(),  # 0表示女，1表示男
    'category': [None] * num_users,
    'price': [None] * num_users,
    'description': [None] * num_users
}

categories = ['electronics', 'books', 'clothing', 'home', 'sports', 'toys']
descriptions = [
    'Smartphone with 6GB RAM', 'Fiction book', '4K Television', 'T-shirt', 'Basketball', 'Lego set',
    'Laptop with 16GB RAM', 'Non-fiction book', 'Washing machine', 'Jacket', 'Football', 'Action figure',
    'Tablet with 8GB RAM', 'Science fiction book', 'Refrigerator', 'Jeans', 'Tennis racket', 'Doll house',
    'Camera with 20MP', 'Historical book', 'Microwave oven', 'Sweater', 'Running shoes', 'Board game',
    'Smartwatch', 'Biography book', 'Vacuum cleaner', 'Shirt', 'Yoga mat', 'Puzzle',
    'Gaming console', 'Mystery book', 'Blender', 'Dress', 'Bicycle', 'Toy car',
    'Headphones', 'Fantasy book', 'Air fryer', 'Coat', 'Helmet', 'Toy train',
    'E-reader', 'Romance book', 'Coffee maker', 'Skirt', 'Dumbbells', 'Stuffed animal',
    'Fitness tracker', 'Thriller book', 'Dishwasher', 'Pants', 'Soccer ball', 'Card game',
    'Portable speaker', 'Cookbook', 'Toaster', 'Shorts', 'Baseball bat', 'Art supplies',
    'Drone', 'Adventure book', 'Electric kettle', 'Scarf', 'Golf clubs', 'Building blocks',
    'Projector', 'Poetry book', 'Slow cooker', 'Hat', 'Skateboard', 'Craft kit'
]

item_data = {
    'node': items,
    'type': ['item'] * num_items,
    'age': [None] * num_items,
    'gender': [None] * num_items,
    'category': [random.choice(categories) for _ in range(num_items)],
    'price': np.round(np.random.uniform(10, 1000, size=num_items), 2).tolist(),
    'description': random.sample(descriptions, num_items)
}

nodes = pd.concat([pd.DataFrame(user_data), pd.DataFrame(item_data)], ignore_index=True)

relations = ['buys', 'views', 'rates']
edges_data = {
    'source': random.choices(users, k=200),
    'target': random.choices(items, k=200),
    'relation': random.choices(relations, k=200)
}

edges = pd.DataFrame(edges_data)

# 保存节点和边数据到CSV文件
nodes.to_csv('nodes_1.csv', index=False)
edges.to_csv('edges_1.csv', index=False)

print("数据已生成并保存到nodes_1.csv和edges_1.csv中。")