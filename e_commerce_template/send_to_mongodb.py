# import json
# from pymongo import MongoClient

# def send_to_mongogb():

#     # MongoDB 连接配置
#     mongo_uri = "mongodb+srv://zhaoxuanjermaine:p0v8SGEj6qOy8oQ8@cluster0.jydo34v.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"  # 替换为您的 MongoDB 连接 URI
#     database_name = "cluster0"       # 替换为您的数据库名称
#     collection_name = "user_items_1"   # 替换为您的集合名称

#     # 读取 JSON 文件
#     with open('/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/items_of_jermainezhao.json', 'r') as file:
#         data = json.load(file)

#     # 连接到 MongoDB
#     client = MongoClient(mongo_uri)
#     db = client[database_name]
#     collection = db[collection_name]

#     # 插入数据到 MongoDB
#     if isinstance(data, list):
#         collection.insert_many(data)
#     else:
#         collection.insert_one(data)

#     print("数据已成功保存到 MongoDB 数据库")

#     # 关闭 MongoDB 连接
#     client.close()






import json
from pymongo import MongoClient


def send_to_mongodb(mongo_uri, database_name, collection_name,json_name):
    # MongoDB 连接配置
    # mongo_uri = "mongodb+srv://zhaoxuanjermaine:p0v8SGEj6qOy8oQ8@cluster0.jydo34v.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"  # 替换为您的 MongoDB 连接 URI
    # database_name = "cluster0"       # 替换为您的数据库名称
    # collection_name = collection_name_1   # 替换为您的集合名称

    # 读取 JSON 数据
    with open(json_name, 'r') as file:
        data = json.load(file)

    # 连接到 MongoDB
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    # 查找现有的最大 item_id
    max_item_id = -1
    all_documents = collection.find()
    for document in all_documents:
        for item in document["items"]:
            if "item_id" in item and item["item_id"] > max_item_id:
                max_item_id = item["item_id"]

    # 更新 data 中的 item_id
    for item in data["items"]:
        max_item_id += 1
        item["item_id"] = max_item_id

    # 插入数据到 MongoDB
    collection.insert_one(data)

    print("数据已成功保存到 MongoDB 数据库")

    # 关闭 MongoDB 连接
    client.close()

# collection_name = "user_items_1"
# database_name = "cluster0"  
# mongo_uri = "mongodb+srv://zhaoxuanjermaine:p0v8SGEj6qOy8oQ8@cluster0.jydo34v.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"  # 替换为您的 MongoDB 连接 URI
# json_name = '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/items_of_jermainezhao.json'

# # 调用函数
# send_to_mongodb(mongo_uri, database_name, collection_name,json_name)