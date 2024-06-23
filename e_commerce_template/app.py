from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import subprocess
import logging

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/uploads'  # 确认路径存在并且可写
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

client = MongoClient('mongodb+srv://zhaoxuanjermaine:p0v8SGEj6qOy8oQ8@cluster0.jydo34v.mongodb.net/')
db = client['cluster0']
collection = db['user_items_1']

@app.route('/')
def home():
    return render_template('store_gallery.html')

@app.route('/api/items', methods=['GET','POST'])
def get_items():
    user_id = "jermainezhao"
    user_data = collection.find({"user_id": user_id})
    items_list = []
    for document in user_data:
        if "items" in document:
            for item in document["items"]:
                items_list.append({
                    "name": item["item_name"],
                    "image_url": item["amazon_image_url"],
                    "estimated_price": item["estimated_price"],
                    "description": item["description"],
                    "original_img": item["original_image_base_64"],
                    "item_id": item["item_id"],
                })
    return jsonify(items_list)

@app.route('/api/items/<int:item_id>', methods=['GET','POST'])
def get_item(item_id):
    user_id = "jermainezhao"
    user_data = collection.find({"user_id": user_id})
    for document in user_data:
        if "items" in document:
            for item in document["items"]:
                if item["item_id"] == item_id:
                    return jsonify({
                        "item_id": item["item_id"],
                        "name": item["item_name"],
                        "image_url": item["amazon_image_url"],
                        "estimated_price": item["estimated_price"],
                        "description": item["description"],
                        "original_img": item["original_image_base_64"]
                    })
    return jsonify({"error": "Item not found"}), 404

@app.route('/api/upload', methods=['GET','POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.debug(f"File saved to {filepath}")
        
        try:
            result = subprocess.run(['python3', '/Users/jermainezhao/CalHack_Hand_to_Hand/Hand_to_Hand/e_commerce_template/main.py', filepath], capture_output=True, text=True, check=True)
            logging.debug(f"subprocess output: {result.stdout}")
            return jsonify({"message": "File uploaded and processed", "output": result.stdout}), 200
        except subprocess.CalledProcessError as e:
            logging.error(f"subprocess error: {e.stderr}")
            return jsonify({"error": "File processing failed", "details": e.stderr}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)