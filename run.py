from flask import Flask, app, render_template, request, redirect, flash, url_for, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model

init_Base64 = 21

ann_model = load_model('models/my_keras_model.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        draw = request.form['url']
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded))
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        cv2.imwrite("./static/genda.jpg" ,image)
        
        # Neural Networks
        img = (np.expand_dims(resized, 0))
        ann_result = ann_model.predict(img/255.0)


    return render_template('index.html', url1 = "./static/genda.jpg",
                           ann_result = np.argmax(ann_result, axis=1)[0], 
                           ann_probs=ann_result[0].round(2), )

if __name__=="__main__":
    app.run(port="8000", debug=True)
