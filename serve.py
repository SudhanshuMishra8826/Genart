from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import argparse
import imutils
import time
import cv2
import json
import requests
import os
import numpy as np
import base64

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug= False


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/create', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(filename)

            addr = 'http://localhost:5000'
            test_url = addr + '/api'
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}
            img = cv2.imread(filename)
            option ={ 
                'option':request.form['second']
            }
            _, img_encoded = cv2.imencode('.jpg', img)
            response = requests.post(test_url, data=img_encoded.tostring(), headers=headers,params=option)
            data = response.json()

            
            return render_template("res.html",data = data)

    return render_template("create.html")


@app.route("/api", methods=["POST"])
def api():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('img.jpg', image)
    if(request.args.get('option')=='1'):
        style="./instance_norm/candy.t7"
    elif(request.args.get('option')=='2'):
        style="./instance_norm/composition_vii.t7"
    elif(request.args.get('option')=='3'):
        style="./instance_norm/feathers.t7"
    elif(request.args.get('option')=='4'):
        style="./instance_norm/la_muse.t7"
    elif(request.args.get('option')=='5'):
        style="./instance_norm/mosaic.t7"
    elif(request.args.get('option')=='6'):
        style="./instance_norm/starry_night.t7"
    elif(request.args.get('option')=='7'):
        style="./instance_norm/the_scream.t7"
    elif(request.args.get('option')=='8'):
        style="./instance_norm/the_wave.t7"
    elif(request.args.get('option')=='9'):
        style="./instance_norm/udnie.t7"


    net = cv2.dnn.readNetFromTorch(style)
    image = cv2.imread('img.jpg')
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)

    output = net.forward()

    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680

    # output /= 255.0
    output = output.transpose(1, 2, 0).astype(int)
    _, img_encoded = cv2.imencode('.jpg', output)

    output_b64 = base64.b64encode(img_encoded)

    # cv2.imwrite('img3.jpg', output)
    # cv2.imshow("Input", image)
    # cv2.imshow("Output", output)
    result = {'output_label':1, 'image_data': output_b64.decode()}
    return jsonify(result)

    
if __name__=="__main__":
    app.run()
