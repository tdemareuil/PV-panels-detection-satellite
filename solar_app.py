import os
import sys
import flask
import requests
import skimage
import numpy as np
import matplotlib.pyplot as plt
import json

import solar

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from tensorflow.keras import optimizers
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import img_to_array
from model import RetinaNetParking
import tensorflow as tf
from PIL import Image
import json
import io

from flask_restful import Resource, Api
from flask import Response

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# This optional command forces to output all the contents of the stdout buffer
# sys.stdout.flush()

model = None


################################################################
#  Paths to dependencies
################################################################

# Directory of the mrcnn library
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
SOLAR_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs", "mask_rcnn_solar_0025.h5")

# Path to results directory (TBD)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


################################################################
#  App
################################################################

app = flask.Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/', methods=["POST", "GET"])
def home():
    """ Simple view function run when the route specified above 
        is requested by user. Displays home page based on html template.
    """
    return flask.render_template("index.html")

@app.route('/success', methods=["POST", "GET"])
def predict():

    # Check if the post request has either an uploaded or default image
    if request.method == 'POST':
        
        if flask.request.files['file']:
            image = flask.request.files['file']
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload.png'))
            # If we save the file under its original filename, we should
            # run it through secure_filename for security before saving,
            # with: filename = secure_filename(image.filename)
        
        elif flask.request.form.get('default_image'):
            image = flask.request.form.get('default_image', None)
            if image is None:
                flask.flash('No file uploaded')
                return redirect(request.url)
            
        # Read the image as a numpy array and resize it
        image = skimage.io.imread(image)
        image, window, scale, padding, crop = utils.resize_image(image,
                                                min_dim=inference_config.IMAGE_MIN_DIM,
                                                min_scale=inference_config.IMAGE_MIN_SCALE,
                                                max_dim=inference_config.IMAGE_MAX_DIM,
                                                mode=inference_config.IMAGE_RESIZE_MODE)

        # Run detection
        yhat = model.detect([image], verbose=0)[0]
        n_solar = yhat['masks'].shape[2]
        print("Nb of solar arrays detected: ", n_solar)
        pv_surface = compute_mask_to_surface(yhat['masks'])
        pv_size_annotation = [str(round(x,1))+"m^2" for x in pv_surface]

        # Save prediction in static folder
        visualize.display_instances(image, yhat['rois'], yhat['masks'],
                                    yhat['class_ids'], class_names=['BG', 'solar array'],
                                    yhat['scores'], captions=pv_size_annotation,
                                    title="{} PV arrays detected".format(n_solar),
                                    plot=False, save="static/result.png")
        
        # Save mask in static folder
        mask = yhat['masks'].astype(int)
        mask = mask[...,0:1]+mask[...,1:2]
        mask = mask.squeeze()
        skimage.io.imsave("static/mask.png", mask)
   
        # Save detection information in static folder
        data = {}
        data["Successful detection"] = True
        data["Number of detected solar arrays"] = str(n_solar)
        data["Total surface of detected solar arrays"] = "≈"+str(round(pv_surface.sum(),1))+"m^2"
        data["Detection confidence level"] = "≈"+str(round(yhat['scores'].mean()*100,1))+"%"
        # Save as json
        with open('static/detection_info.json', 'w') as outfile:
            json.dump(data, outfile)
        # Save as Excel (for non-tech people)
        df = pd.DataFrame(data, index=[0]).T
        df.to_excel('static/detection_info.xlsx', header=False)
        
        return redirect(url_for('upload')) # not sure if we need this
    
    return render_template('success_nst.html')

if __name__ == '__main__':
# the app will run only if it's called as main,
# i.e. not if you import the app in another code
    print("...Loading model and starting server\n",
          "...please wait until server has fully started")
     
    # Create model in inference mode
    inference_config = solar.InferenceConfig()
    
    # Recreate the model in inference mode
    global model
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=SOLAR_WEIGHTS_PATH)
    # Load trained weights
    model.load_weights(SOLAR_WEIGHTS_PATH, by_name=True)
    print("Loaded model:", MODEL_NAME)
    
    # Run app
    app.run(host='0.0.0.0', port='5000', debug=True)