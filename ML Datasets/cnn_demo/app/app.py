from flask import Flask, request, jsonify
from flask import render_template
from flask import request,send_from_directory
from PIL import Image
import base64
import io
from flask_cors import CORS, cross_origin
import time
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
target_size = (30,30)#for inference


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/infer')
def infer():
    return render_template('infer.html')

@app.route('/draw')
def draw():
    return render_template('index.html')



@app.route('/api/v1/add_doodle', methods=['POST'])
@cross_origin()
def process_doodle():
    b64_text = request.form.get('image')
    img_data = str.encode(b64_text.split(',')[-1])#pull just the data off the b64 header from the URL image data
    img = None
    t = int(time.time())
    fn = os.path.join(f'data',f'{t}.jpg')
    with open(fn, "wb") as fh:
        img = base64.decodebytes(img_data)
        fh.write(img)
    
    if img is not None:
        img = Image.open(io.BytesIO(img))


        return jsonify({
                    'msg': 'success', 
                    'size': [img.width, img.height], 
                    'format': img.format
            })
    else:
        return jsonify({
                    'msg': 'failed - b64 image not sent.'
            }) 

@app.route('/api/v1/infer_doodle', methods=['POST'])
@cross_origin()
def infer_doodle():
    b64_text = request.form.get('image')
    img_data = str.encode(b64_text.split(',')[-1])#pull just the data off the b64 header from the URL image data
    img = None
    try: 
        img = base64.decodebytes(img_data)
    except:
        pass
    if img is not None:
        
        model = keras.models.load_model('shapes_v1.keras')
        img = Image.open(io.BytesIO(img))
        img = img.convert('L')
        img = img.resize(target_size, Image.NEAREST)
        input_arr = tf.keras.utils.img_to_array(img)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        result = model.predict(input_arr)[0].tolist()
        result = [ round(elem,2) for elem in result ]
        print(result)

        return jsonify({
                    'result':result,
                    'msg': 'success', 
                    'size': [img.width, img.height], 
                    'format': img.format
            })
    else:
        return jsonify({
                    'msg': 'failed - b64 image not sent.'
            }) 
# endpoint route for static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)
    
    
if __name__ == '__main__':
    app.run(debug=True)