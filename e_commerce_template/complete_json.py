import json

def complete_json(user_id):
    # 读取原始 JSON 文件
    with open('/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/Merged_Item_Features.json', 'r') as file:
        data = json.load(file)

    # 初始化新的字典结构
    updated_data = {"user_id": user_id, "items": []}

    # 遍历原始数据并添加新字段
    for index, (key, value) in enumerate(data.items()):
        value["item_id"] = index
        value["lowest_price"] = None  # 根据需求初始化
        value["location"] = None      # 根据需求初始化
        value["browsing_times"] = 0   # 根据需求初始化
        value["if_sold"] = False      # 根据需求初始化
        value["sold_price"] = None    # 根据需求初始化
        updated_data["items"].append(value)

    # 将更新后的数据写入新的 JSON 文件
    with open(f'items_of_{user_id}.json', 'w') as file:
        json.dump(updated_data, file, indent=4)

    print(f"JSON file saved as 'items_of_{user_id}.json'")