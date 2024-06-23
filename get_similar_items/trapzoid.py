from PIL import Image

def trapezoid_transform(image_path, output_path):
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: The file at {image_path} was not found.")
        return
    except OSError:
        print(f"Error: The file at {image_path} is not a valid image.")
        return

    width, height = image.size

    # 定义新的四个角点，形成梯形
    new_corners = [
        width * 0.2, 0,          # 左上角稍微向右移动
        width * 0.8, 0,          # 右上角稍微向左移动
        width, height,           # 右下角不变
        0, height                # 左下角不变
    ]

    try:
        # 生成新的图像
        transformed_image = image.transform(
            (width, height),
            Image.QUAD,
            data=new_corners,
            resample=Image.BICUBIC
        )

        # 将图像转换为 RGB 模式以保存为 JPEG
        if transformed_image.mode == 'RGBA':
            transformed_image = transformed_image.convert('RGB')

        # 保存图像
        transformed_image.save(output_path, format='JPEG')
        transformed_image.show()
    except Exception as e:
        print(f"An error occurred: {e}")

# 使用示例
trapezoid_transform('/Users/jermainezhao/Get_similar_items/amazon_items_visualized.png', 'transformed_image.jpg')