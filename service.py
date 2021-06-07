from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def hello_world():
    print('[log] recieved query')
    return 'Hello, World!'

app.run(port=9000, threaded=False, host="127.0.0.1")