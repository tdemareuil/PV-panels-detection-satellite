import flask
import requests
import skimage
import numpy as np
import json
import tensorflow as tf
import pandas as pd
import geojson
import rasterio as rio
import glob
import sys
import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


################################################################
#  Paths to dependencies
################################################################

# Directory of the mrcnn library
ROOT_DIR = os.path.abspath("../")

# Import Mask R-CNN and solar.py
import solar
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
MODEL_NAME = "mask_rcnn_solar_0025.h5"
SOLAR_WEIGHTS_PATH = os.path.join(ROOT_DIR, "pvpanels", "models", MODEL_NAME)


################################################################
#  App
################################################################

app = flask.Flask(__name__)

# app.config['APP_ROOT'] = os.path.dirname(os.path.abspath(__file__))
app.config['APP_ROOT'] = app.root_path
app.config['APP_STATIC'] = os.path.join(app.config['APP_ROOT'], 'static')
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['APP_ROOT'], 'inference')
app.config['MAX_CONTENT_LENGTH'] = 8 * 2048 * 2048
app.config['SECRET_KEY'] = "eb3373cdd21535d1fd204e13"
# Random key above was generated with: os.urandom(12).hex()


# Define the home page
@app.route('/', methods=["POST", "GET"])
def home():
    return flask.render_template("index.html")


# Define the success page
@app.route('/success', methods=["POST", "GET"])
def predict():

    # Check if the post request has either an uploaded or default image
    if flask.request.method == 'POST':
        
        # Option 1: Get user-uploaded image
        if flask.request.files.get('file'):
            image = flask.request.files['file']
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload.png'))
            # If we had chosen to save the file under its original filename, for
            # security we would have run: filename = secure_filename(image.filename)
        
        # Option 2: Get chosen default image
        elif flask.request.form.get('test'):
            image = os.path.join(app.config['APP_STATIC'], flask.request.form.get('test'))
        
        # Option 3: Get image under chosen coordinates through Google API
        elif flask.request.form.get('lat'):
            LAT = flask.request.form.get('lat')
            LNG = flask.request.form.get('lng')
            SIZE = flask.request.form.get('size')
            # If size is larger than 200, redirect to batch processing page
            if SIZE:
                if int(SIZE) > 200:
                    # flask.flash('Your query covers a large area. Processing might ' +
                    #     'take some time (several minutes). Please come back in a ' +
                    #     'while to check results!')
                    return flask.redirect(flask.url_for('predict_batch',
                                                        lat=LAT, lng=LNG, size=SIZE))
            # Else, continue processing on 'success' page
            else:
                try:
                    solar.call_google_earthenginesolar_API(
                        LAT, LNG, ID=1, RADIUS=100, 
                        API_KEY='AIzaSyA-B9aw7KZlLwJHpQveCQhtOSjm9Omqniw', 
                        output_path=app.config['UPLOAD_FOLDER'],
                        compute_RGB_IMG=True,
                        compute_Flux_IMG=False,
                        compute_Elevation_IMG=False)
                    image = os.path.join(app.config['UPLOAD_FOLDER'], '1_RGB.tif')
                except:
                    flask.flash('No high-resolution satellite images available ' +
                        'for this location. Please try other coordinates.')
                    return flask.redirect(flask.url_for('home'))

        # Read the image as a numpy array and resize it
        image = skimage.io.imread(image, plugin='matplotlib')        
        if image.shape[0]<=512 and image.shape[1]<=512:
            image, *_ = utils.resize_image(image, min_dim=512, max_dim=512, 
                                                min_scale=2.0, mode="square")
        elif image.shape[0]>1024 or image.shape[1]>1024:
            image, *_ = utils.resize_image(image, min_dim=2048, max_dim=2048, 
                                                min_scale=2.0, mode="square")
        else:
            image, *_ = utils.resize_image(image, min_dim=1024, max_dim=1024, 
                                                min_scale=2.0, mode="square")
        print("--> Image loaded and pre-processed - going to detection...")

        # Run detection
        with graph.as_default():
            yhat = model.detect([image], verbose=0)[0]
        n_solar = yhat['masks'].shape[2]
        pv_surface = solar.compute_mask_to_surface(yhat['masks'])
        # pv_size_annotation = [str(round(x,1))+"m^2" for x in pv_surface]
        print("--> Detection successful - now saving results for display...")

        # Save prediction in inference folder
        if n_solar != 0:
            solar.display_instances(image, yhat['rois'], yhat['masks'],
                                yhat['class_ids'], ['BG', 'solar array'],
                                yhat['scores'], plot=False,
                                # caption = pv_size_annotation,
                                save=os.path.join(app.config['UPLOAD_FOLDER'], "result.png"))
        else:
            skimage.io.imsave(os.path.join(app.config['UPLOAD_FOLDER'], "result.png"), image)

        # Save mask in inference folder
        mask = yhat['masks'].astype(int)
        mask = mask.sum(axis=2)
        skimage.io.imsave(os.path.join(app.config['UPLOAD_FOLDER'], "mask.png"), mask)
   
        # Save detection summary
        data = {}
        data["Successful detection"] = True
        data["Number of detected solar arrays"] = str(n_solar)
        data["Total surface of detected solar arrays"] = "≈ {:.1f}m^2".format(pv_surface.sum())
        if n_solar != 0:
            data["Detection confidence level"] = "≈ {:.1%}".format(yhat['scores'].mean())
        else: data["Detection confidence level"] = "n.a."
        # Save as json
        # with open(os.path.join(app.config['UPLOAD_FOLDER'],
        #                        "detection_summary.json"), 'w') as outfile:
        #     json.dump(data, outfile)
        # Save as Excel (for non-tech people)
        # df = pd.DataFrame(data, index=[0]).T
        # df.to_excel(os.path.join(app.config['UPLOAD_FOLDER'],
        #                                     "detection_summary.xlsx"), header=False)
        print("--> Saving done - showing success page.\n")

    return flask.render_template('success.html', data=data)


# Define the batch processing success page
@app.route('/success/batch', methods=["POST", "GET"])
def predict_batch():
    # Get back arguments (coordinates) passed from previous page
    LAT = float(flask.request.args['lat'])
    LNG = float(flask.request.args['lng'])
    SIZE = int(flask.request.args['size'])
    flask.session.pop('_flashes', None) # Clear session
    print("--> Loading images corresponding to the requested area")

    # Build coordinates of all images in the required area
    print("Filtering area...")
    coordinates = solar.get_area_coordinates(LAT, LNG, SIZE)

    # Filter coordinates based on whether there is a building around (50m)
    coordinates = [coords for coords in coordinates if solar.check_around(coords[0],
                                                            coords[1], 90, 'building')]
    n_images = len(coordinates)
    print("{} images selected...".format(n_images))
    if n_images == 0:
        flask.flash('No buildings in this area. Please try other coordinates.')
        return flask.redirect(flask.url_for('home'))

    # Query images corresponding to the filtered list of coordinates
    for index, coords in enumerate(coordinates):
        try:
            solar.call_google_earthenginesolar_API(
                            coords[0], coords[1], ID=index+1, RADIUS=100, 
                            API_KEY='AIzaSyA-B9aw7KZlLwJHpQveCQhtOSjm9Omqniw', 
                            output_path=app.config['UPLOAD_FOLDER'],
                            compute_RGB_IMG=True,
                            compute_Flux_IMG=False,
                            compute_Elevation_IMG=False)
            image = os.path.join(app.config['UPLOAD_FOLDER'],
                                             '{}_RGB.tif'.format(index+1))
            print("Image {}/{} loaded...".format(index+1, n_images))
        except:
            flask.flash('No high-resolution satellite images available \
                for this location. Please try other coordinates or area size.')
            return flask.redirect(flask.url_for('home'))
    print("\n--> All images loaded and pre-processed - going to detection")

    # Get affine transformation of images for conversion from pixel space to crs
    with rio.open(os.path.join(app.config['UPLOAD_FOLDER'], '1_RGB.tif')) as src:
        pixel_transform = src.transform
        crs = src.crs
    
    # Perform detection over all images
    polygons_list = []
    total_solar = 0
    total_surface = 0
    total_confidence = 0
    for index, image in enumerate(sorted(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'],
                                                                '*.tif')))[0:n_images]):
        # Read the image as a numpy array and resize it
        image = skimage.io.imread(image, plugin='matplotlib')
        image, *_ = utils.resize_image(image, min_dim=2048, max_dim=2048, 
                                              min_scale=2.0, mode="square")
        # Run detection
        with graph.as_default():
            yhat = model.detect([image], verbose=0)[0]
        n_solar = yhat['masks'].shape[2]
        pv_surface = solar.compute_mask_to_surface(yhat['masks']).sum()
        pv_confidence = yhat['scores'].mean()*100

        # Convert mask to polygons and reproject to espg:4326
        polygons_image = solar.mrcnn_masks_to_polygons(yhat['masks'],
                                                     pixel_transform, crs)
        polygons_list.extend(polygons_image)

        # Store detection information
        total_solar += n_solar
        total_surface += pv_surface
        total_confidence += pv_confidence
        print("Image {}/{} analysed...".format(index+1, n_images))
    print("\n--> Detection successful - now saving results for display")
 
    # Convert and save full polygons list as geojson in inference folder
    polygons_list = [geojson.Feature(geometry=poly) for poly in polygons_list]
    polygons_list = geojson.FeatureCollection(polygons_list)   
    with open(os.path.join(app.config['UPLOAD_FOLDER'],
                                         'detected_polygons.geojson'), 'w') as f:
        geojson.dump(polygons_list, f)

    # Save detection summary in inference folder
    data = {}
    data["Successful detection"] = True
    data["Area size coordinates"] = "{}m^2".format(SIZE)
    data["Area coordinates (center)"] = "({:.4f},{:.4f})".format(LAT, LNG)
    data["Number of detected solar arrays"] = str(total_solar)
    data["Total surface of detected solar arrays"] = "≈ {:.1f}m^2".format(total_surface)
    if total_solar != 0:
        data["Average confidence level"] = "≈ {:.1%}".format(total_confidence/(int(n_images)*100))
    else: data["Average confidence level"] = "n.a."
    # Save as json
    # with open(os.path.join(app.config['UPLOAD_FOLDER'],
    #                        "detection_summary.json"), 'w') as outfile:
    #     json.dump(data, outfile)
    # Save as Excel (for non-tech people)
    df = pd.DataFrame(data, index=[0]).T
    df.to_excel(os.path.join(app.config['UPLOAD_FOLDER'],
                                        "detection_summary.xlsx"), header=False)
    print("\n--> Saving done - showing success page.\n")

    return flask.render_template('success_batch.html',
                                 data=data, lat=LAT, lng=LNG, polygons=polygons_list)


# This last route is just to make sure flask will accept displaying images 
# from the inference directory
@app.route('/inference/<path:filename>')
def display_file(filename):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)


# Finally, the script that builds the model and launches the app
if __name__ == '__main__':
    # App will run only if called as main, i.e. not if imported in another code
    print("\n...Loading model and starting server...\n",
          "       ...please wait until server has fully started...\n")
     
    # Build the model in inference mode
    inference_config = solar.InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=SOLAR_WEIGHTS_PATH)
    # Load trained weights
    model.load_weights(SOLAR_WEIGHTS_PATH, by_name=True)

    graph = tf.get_default_graph() 
    print("Loaded model:", MODEL_NAME)
    print("Model runs on http://0.0.0.0:5000/\n")

    # Run app
    app.run(host='0.0.0.0', port='5000', debug=True)
