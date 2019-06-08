import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

from flask import Flask, request, redirect
from flask import render_template
from src.models import build_cnn_extractor
from src.database_large import Database
from keras.models import load_model
import keras.backend as K
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

# global database
triplet_db = None
non_triplet_db = None


def init_triplet_database():
    K.clear_session()
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

    return triplet_db


def init_non_triplet_database():
    K.clear_session()
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

    return non_triplet_db


# init_non_triplet_database()
def validate(req):
    if 'file' not in req.files:
        print('No file part')
        return False

    file = req.files['file']

    if file.filename == '':
        print('No selected file')
        return False

    return True


def query(db, file):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    abs_img_path = os.path.join(os.path.dirname(__file__), os.path.relpath(img_path, '..'))
    file.save(abs_img_path)

    num_result = int(request.form['num-result'])
    use_kmeans = False
    if 'use-kmeans' in request.form:
        use_kmeans = True
    query_image = db.load_image(abs_img_path)
    start = time()
    result_path = db.query(query_image, use_kmeans=use_kmeans, num_results=num_result, return_path=True)
    end = time()
    query_time = end - start
    client_result_path = []
    for path in result_path:
        image_name = ntpath.basename(path)
        image_path = os.path.join(RESULT_FOLDER, image_name)
        abs_image_path = os.path.join(os.path.dirname(__file__), os.path.relpath(image_path, '..'))
        shutil.copyfile(path, abs_image_path)
        client_result_path.append(image_path)

    return img_path, client_result_path, query_time


@app.route("/", methods=['GET'])
def index():
    return redirect("/triplet")


@app.route("/triplet", methods=['GET', 'POST'])
def triplet():
    global triplet_db
    # if triplet_db is None:

    if request.method == 'GET':
        triplet_db = init_triplet_database()
        return render_template('query.html')

    if request.method == 'POST':
        if not validate(request):
            redirect(request.url)

        file = request.files['file']
        img_path, client_result_path, query_time = query(triplet_db, file)

        return render_template('query.html', img_path=img_path, result_path=client_result_path, query_time=query_time)


@app.route("/non-triplet", methods=['GET', 'POST'])
def non_triplet():
    global non_triplet_db
    # if non_triplet_db is None:

    if request.method == 'GET':
        non_triplet_db = init_non_triplet_database()
        return render_template('query.html')

    if request.method == 'POST':
        if not validate(request):
            redirect(request.url)

        file = request.files['file']
        img_path, client_result_path, query_time = query(non_triplet_db, file)

        return render_template('query.html', img_path=img_path, result_path=client_result_path, query_time=query_time)


if __name__ == '__main__':
    app.secret_key = 'any random string'
    app.run(debug=True)
