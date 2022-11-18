import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('ECG.h5')


@app.route('/')
def about():
    return render_template('about.html')


@app.route('/about')
def home():
    return render_template('about.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x), axis=1)
        index = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction',
                 'Premature Ventricular Contractions', 'Right Bundle Branch Block', 'Ventricular Fibrillation']
        text = "Arrhythmia Type:"+str(index[pred[0]])
        return text


if __name__ == "__main__":
    app.run(debug=False)
