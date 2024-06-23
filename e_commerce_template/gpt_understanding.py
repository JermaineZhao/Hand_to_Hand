import openai
import base64
import requests
import os
import json
import re

openai.api_key = ""

# # Read original json file
# with open('/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/All_item_features.json', 'r') as file:
#     data = json.load(file)

def extract_json_string(input_string):
    start_index = input_string.find('{')
    end_index = input_string.rfind('}') + 1
    if start_index != -1 and end_index != -1:
        return input_string[start_index:end_index]
    return None

def generate_details(data):
    new_data = {}
    for key, item in data.items():

        response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                        "type": "text",
                        "text": f"The first image is the item image on amazon(new). The second image is the picture of the same item but second-hand.\
                                The key words of the items are: {item['key_words']}.\
                                The new item price on Amazon is ${item['average_price']}.\
                                Please provide me: \
                                1. item_name: Combine the key words and the images, give me a detailed item name(about 4-8 words); \
                                2. condition: Compare the Amazon picture and real-life second-hand picture, tell me the condition of the item in 5 categories(like new, very good, good, acceptable, not so well); \
                                3. estimated_price: According to the Amazon price and item's condition, give me an estimated price of the second-hand item beginning with USD sign(According the condition, the estimated price cannot exceed 70% of the Amazon price.); \
                                4. description: generate a detailed description about the item (including condition and usage). \
                                Please return me in JSON format with 4 keys and 4 values.\
                                "
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{item['amazon_image_url']}"
                        }
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{item['original_image_base_64']}"
                        }
                        },
                    ],
                    }
                ],
                max_tokens=400,
                )
        
        answer = response.choices[0].message.content
        json_answer = extract_json_string(answer)

        response_data = json.loads(json_answer)

        new_data[key] = {
            "item_name": response_data.get("item_name", ""),
            "condition": response_data.get("condition", ""),
            "estimated_price": response_data.get("estimated_price", 0),
            "description": response_data.get("description", "")
        }
        print(key)
    return new_data

def main3():
    openai.api_key = ""

    # Read original json file
    with open('/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/All_item_features.json', 'r') as file:
        data = json.load(file)
    new_data = generate_details(data)

    with open('additional_data.json', 'w') as file:
        json.dump(new_data, file, indent=4)

    print("Processing complete. Check 'output_data.json' for the results.")

        