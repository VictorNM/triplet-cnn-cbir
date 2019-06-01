import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

from flask import Flask, flash, request, redirect
from flask import render_template
from src.models import build_cnn_extractor
from src.database_large import Database
from keras.models import load_model
import shutil
import ntpath
from time import time


UPLOAD_FOLDER = '../static/upload'
RESULT_FOLDER = '../static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

TRIPLET_EXTRACTOR_PATH = '../models/final/triplet-2019-05-24 21_06_53.h5'
CNN_CLASSIFIER_PATH = '../models/final/2019-05-24 21_06_53.h5'

DATABASE_DIRECTORY = '../database/stanford_online_products'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def init_triplet_database():
    global triplet_db

    triplet_extractor = load_model(os.path.join(
        os.path.dirname(__file__),
        TRIPLET_EXTRACTOR_PATH
    ))
    triplet_db = Database(triplet_extractor, os.path.join(
        os.path.dirname(__file__),
        DATABASE_DIRECTORY
    ))
    triplet_db.load_features_database('features-triplet')
    triplet_db.create_kmeans(2)


def init_non_triplet_database():
    global non_triplet_db
    cnn_classifier = load_model(os.path.join(
        os.path.dirname(__file__),
        CNN_CLASSIFIER_PATH
    ))
    cnn_extractor = build_cnn_extractor(cnn_classifier)
    non_triplet_db = Database(cnn_extractor, os.path.join(
        os.path.dirname(__file__),
        DATABASE_DIRECTORY
    ))
    non_triplet_db.load_features_database('features-non-triplet')
    non_triplet_db.create_kmeans(2)


init_triplet_database()
# init_non_triplet_database()


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        abs_img_path = os.path.join(os.path.dirname(__file__), os.path.relpath(img_path, '..'))
        file.save(abs_img_path)

        query_image = triplet_db.load_image(abs_img_path)
        start = time()
        result_path = triplet_db.query(query_image, use_kmeans=False, num_results=12, return_path=True)
        end = time()
        query_time = end - start
        client_result_path = []
        for path in result_path:
            image_name = ntpath.basename(path)
            image_path = os.path.join(RESULT_FOLDER, image_name)
            abs_image_path = os.path.join(os.path.dirname(__file__), os.path.relpath(image_path, '..'))
            shutil.copyfile(path, abs_image_path)
            client_result_path.append(image_path)

        return render_template('index.html', img_path=img_path, result_path=client_result_path, query_time=query_time)