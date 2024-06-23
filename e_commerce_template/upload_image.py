import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import os
import requests
import base64
import json
import boto3
from botocore.exceptions import NoCredentialsError

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        url = f"https://{bucket}.s3.amazonaws.com/{s3_file}"
        return url
    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None


def find_and_click_image(driver, template_path, click_offset=(0, 0)):
    while True:
        # 截图页面
        driver.save_screenshot("screenshot.png")
        screen = cv2.imread("screenshot.png")

        # 读取模板图像
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]

        # 转换截图为灰度图
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # 在截图中找到模板
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        if loc[0].size > 0:
            for pt in zip(*loc[::-1]):
                # 点击图像中心
                x = pt[0] + w // 2 + click_offset[0]
                y = pt[1] + h // 2 + click_offset[1]

                # 将坐标转换为相对于当前窗口的位置
                window_size = driver.get_window_size()
                screen_size = screen.shape
                scale_x = window_size['width'] / screen_size[1]
                scale_y = window_size['height'] / screen_size[0]

                x_scaled = int(x * scale_x)
                y_scaled = int(y * scale_y)

                actions = ActionChains(driver)
                actions.move_by_offset(x_scaled, y_scaled).click().perform()
                actions.move_by_offset(-x_scaled, -y_scaled).perform()
                time.sleep(1) # speedup
                return
        else:
            driver.get('https://www.amazon.com/b?ie=UTF8&node=17387598011')
            time.sleep(0.3) # speedup

def get_prices_and_titles(driver): 
    # 获取搜索结果的HTML
    html_content = driver.page_source

    # 使用BeautifulSoup解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 找到所有包含价格的元素
    product_elements = soup.select('div.product-container div.product-grid article.cellContainer')

    # 提取前八个商品的名称和价格
    products = []
    for product_element in product_elements[:8]:
        title_element = product_element.select_one('h5.a-size-base-plus.a-spacing-mini.a-color-base.a-text-bold')
        price_whole_element = product_element.select_one('span.a-price span.a-price-whole')
        price_fraction_element = product_element.select_one('span.a-price span.a-price-fraction')

        if title_element and price_whole_element and price_fraction_element:
            title = title_element.text.strip()
            price_whole = price_whole_element.text.strip()
            price_fraction = price_fraction_element.text.strip()
            full_price = f"{price_whole}.{price_fraction}"
            products.append((title, full_price))

    for product_element in product_elements[:1]:
        img_tag = product_element.find('img', class_='cellImage')
        if img_tag:
            img_url = img_tag['src']
            products.append(img_url)

    return products

def average_price_without_top_four(products):
    # 过滤掉不符合格式的项
    valid_products = [p for p in products if isinstance(p, tuple) and len(p) == 2]
    
    # 提取价格并转换为浮点数
    prices = [float(price) for _, price in valid_products]
    
    # 检查是否有足够的价格
    if len(prices) <= 4:
        raise ValueError("Not enough products to remove the top four prices")
    
    # 排序价格列表
    prices.sort(reverse=True)
    
    # 移除四个最高的价格
    prices = prices[4:]
    
    # 计算剩下价格的平均值
    average_price = sum(prices) / len(prices)
    
    return average_price

# # 示例列表
# products = [('Braun', '329.94'), ('Braun', '319.94'), ('Braun', '279.94'), ('Braun', '429.99'),
#             ('Braun', '379.99'), ('Braun', '294.99'), ('Braun', '319.99'), ('Braun', '379.94'), 
#             'https://m.media-amazon.com/images/I/711P-WEa+UL._SL400_.jpg']

# # 调用函数并打印结果
# average = average_price_without_top_four(products)
# print("The average price is:", average)

def extract_product_names(products):
    # 过滤掉不符合格式的项，并提取商品名称
    product_names = [p[0] for p in products if isinstance(p, tuple) and len(p) == 2]
    return product_names

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def convert_json(average_price, key_words, amazon_image_url, original_image_url):
    return {
        "average_price": average_price,
        "key_words": key_words,
        "amazon_image_url": amazon_image_url,
        "original_image_base_64": original_image_url
    }

def get_image_paths(folder_path):
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}  # 可以根据需要添加更多扩展名
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

def main2():
    # 设置WebDriver
    driver = webdriver.Chrome()

    # 打开Amazon搜索页面
    driver.get('https://www.amazon.com/b?ie=UTF8&node=17387598011')

    # 模拟点击搜索框并输入内容
    # search_box = driver.find_element(By.ID, "twotabsearchtextbox")
    # search_box.send_keys('eastman e10m')
    # search_box.send_keys(Keys.RETURN)

    time.sleep(3)  # 等待页面加载 # speedup

    # 找到并点击搜索图标
    search_icon_path = "/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/amazon_icon/search.png"
    find_and_click_image(driver, search_icon_path)

    # 等待上传窗口打开
    time.sleep(0.2) # speedup

    # 找到并点击“Upload Image”按钮
    upload_image_icon_path = "/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/amazon_icon/upload_image.png"
    find_and_click_image(driver, upload_image_icon_path)

    # 等待文件选择窗口打开
    time.sleep(0.2) # speedup

    # 模拟文件上传
    folder_path = '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/segmented_objects/items'
    image_paths = get_image_paths(folder_path)

    results = {} # 待会输送到json file

    for i, image_path in enumerate(image_paths):
        upload_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        upload_input.send_keys(os.path.abspath(image_path))
        time.sleep(4)  # 等待上传完成 # speedup
        
        # 获取前八个搜索结果的商品名和价格
        products = get_prices_and_titles(driver)
        # print(f"Products for {image_path}: {products[:8]}")
        average_price = average_price_without_top_four(products)
        # print(f"average price: {average_price}")
        key_words = extract_product_names(products)
        # print(f"key words: {key_words}\n")
        amazon_image_url = products[-1]

        # get s3 url
        local_file = image_path
        s3_file = f"user_1/{os.path.basename(image_path)}"
        bucket = 'jermainezhao'

        orinal_image_url = upload_to_aws(local_file, bucket, s3_file)

        results[i] = convert_json(average_price,key_words,amazon_image_url,orinal_image_url)

        time.sleep(1) 

        # 再次找到并点击搜索图标以重新开始
        find_and_click_image(driver, search_icon_path)
        time.sleep(0.2)  # 等待页面加载 # speedup
        find_and_click_image(driver, upload_image_icon_path)
        time.sleep(0.2) # speedup

    output_json_path = "/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/All_item_features.json"
    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


    # 关闭浏览器
    driver.quit()

    print(f"特征已保存到 {output_json_path}")

# main2()