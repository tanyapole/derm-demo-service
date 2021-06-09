from flask import Flask, send_file, jsonify
import flask as F
import torch
from PIL import Image

app = Flask(__name__)


@app.route('/')
def hello_world():
    # test any response
    print('[log] recieved query')
    return 'Hello, World!'

@app.route("/im_size", methods=["POST"])
def process_image():
    # test sending images
    file = F.request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})

@app.route('/predict', methods=['POST'])
def predict():
    # test sending images
    file = F.request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 
                    'size': [img.width, img.height],
                   'disease': 'acne',
                   'morphology': ['бугорок', 'пузырек']})



app.run(port=9000, threaded=False, host="127.0.0.1")


