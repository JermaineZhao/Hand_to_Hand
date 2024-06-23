import sys
import logging
from extract_all_main import main1
from upload_image import main2
from gpt_understanding import main3
from merge_json import merge
from complete_json import complete_json
from send_to_mongodb import send_to_mongodb

user_id = "jermainezhao"
collection_name = "user_items_1"
database_name = "cluster0"  
mongo_uri = "mongodb+srv://zhaoxuanjermaine:p0v8SGEj6qOy8oQ8@cluster0.jydo34v.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"  # 替换为您的 MongoDB 连接 URI
json_name = '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/items_of_jermainezhao.json'


def main(image_path):
    try:
        main1(image_path)
        print("main1 completed.")
        logging.debug("main1 completed.")
    except Exception as e:
        logging.error(f"main1 failed: {e}")
        raise

    try:
        main2()
        print("main2 completed.")
        logging.debug("main2 completed.")
    except Exception as e:
        logging.error(f"main2 failed: {e}")
        raise

    try:
        main3()
        print("main3 completed.")
        logging.debug("main3 completed.")
    except Exception as e:
        logging.error(f"main3 failed: {e}")
        raise

    try:
        merge()
        print("merge completed.")
        logging.debug("merge completed.")
    except Exception as e:
        logging.error(f"merge failed: {e}")
        raise

    try:
        complete_json(user_id)
        print("complete_json completed.")
        logging.debug("complete_json completed.")
    except Exception as e:
        logging.error(f"complete_json failed: {e}")
        raise

    try:
        send_to_mongodb(mongo_uri, database_name, collection_name,json_name)
        print("Sent to MongoDB!")
        logging.debug("Sent to MongoDB!")
    except Exception as e:
        logging.error(f"send_to_mongodb failed: {e}")
        raise

    print("Everything DONE! HOORAY!!!")
    logging.debug("Everything DONE! HOORAY!!!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        main(image_path)
    else:
        print("No image path provided.")
        logging.error("No image path provided.")