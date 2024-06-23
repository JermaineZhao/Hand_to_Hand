import json

def merge():
    # 假设第一个json文件名为file1.json，第二个json文件名为file2.json
    with open('/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/All_item_features.json', 'r') as f1, open('/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/additional_data.json', 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # 创建一个新的字典来存储合并后的数据
    merged_data = {}

    # 合并两个文件的数据
    for key in data1.keys():
        merged_data[key] = {**data1[key], **data2[key]}

    # 将合并后的数据写入一个新的json文件
    with open('/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/Merged_Item_Features.json', 'w') as mf:
        json.dump(merged_data, mf, indent=4)

    print("合并完成，合并后的文件为 merged_file.json")

# merge()