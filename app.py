from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

model = load_model('Model/detect.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']

vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']


def fetch_calories(prediction):
    try:
        url = f'https://www.google.com/search?&q=calories in {prediction}'
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        print(e)
        return "Calories not found."


def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(" ".join(str(x) for x in y_class))
    res = labels[y]
    return res.capitalize()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction=None)

    file = request.files['image']
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Image Prediction
        prediction = processed_img(file_path)
        if prediction in vegetables:
            category = 'Vegetables'
        else:
            category = 'Fruit'

        # Fetch Calories
        calories = fetch_calories(prediction)

        return render_template('index.html', prediction=prediction, category=category, calories=calories, image_url=file_path)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
