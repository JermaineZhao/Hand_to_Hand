from flask import Flask, render_template, jsonify
import subprocess

app = Flask(__name__, template_folder='.', static_url_path='', static_folder='.')
@app.route('/')
def index():
    # Run the get_both_free.py script
    subprocess.run(["python", "get_both_free.py"])
    
    return render_template('feed-3.html')

@app.route('/get_links')
def get_links():
    with open("location_link.txt", "r") as a_file:
        tbd_link = a_file.read().strip()
    
    with open("human_readable.txt", "r") as b_file:
        todo_content = b_file.read().strip()
    
    return jsonify({
        "tbd_link": tbd_link,
        "todo_content": todo_content
    })

if __name__ == '__main__':
    app.run(port=5502, debug=True)  # 指定不同的端口