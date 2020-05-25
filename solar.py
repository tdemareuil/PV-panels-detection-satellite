import os
import sys
import numpy as np
import skimage
import urllib
import json
import osmnx as ox
import shapely
import pyproj
import imantics
import math
import random
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


################################################################
#  Paths to dependencies
################################################################

# Directory of the mrcnn library
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Path to trained pre-weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR,
                        "pvpanels", "models", "mask_rcnn_coco.h5")
# if not os.path.exists(COCO_WEIGHTS_PATH):
#     utils.download_trained_weights(COCO_WEIGHTS_PATH)

# Path to dataset - change if needed
DATA_DIR = os.path.join(ROOT_DIR, "pvpanels", "data")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
MODELS_DIR = os.path.join(ROOT_DIR, "pvpanels", "models")


################################################################
#  Override Mask R-CNN Configuration class
################################################################

class SolarConfig(Config):
    """Configuration for training on the solar dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "solar"

    # We use a GPU with 16GB memory, which can fit several images.
    # Adjust down if you use a smaller GPU.
    # Batch size is (GPUs * images/GPU).
    # GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Backbone network architecture
    # Supported values are: resnet50, resnet101 (default)
    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + solar array

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1748*2/2 # nb train * (mean nb augments + 1) / nb per GPU
    # Which means that if we have 1748 images and apply on average 1 augmentation
    # per image (sometimes 1, sometimes 0, sometimes 2), we run through our
    # dataset twice per epoch (i.e. we multiply training steps by 2)

    # Use small validation steps since the epoch is small
    VALIDATION_STEPS = 219*2/2 # nb val * (mean nb augments + 1) / nb per GPU

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # You can also choose to not exclude based on confidence. Since we have 
    # 2 classes, 0.5 is the minimum anyway as it picks between solar and BG.
    # DETECTION_MIN_CONFIDENCE = 0

    # Input image resizing. Image size must be dividable by 2 at least 6 times 
    # to avoid fractions when downscaling and upscaling.
    IMAGE_RESIZE_MODE = "crop" # apply random crops on images > 512px
    IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # square anchor side in pixels

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.9

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 1000

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3 (or as speicified). You can increase the number of proposals 
    # by adjusting the RPN NMS threshold. Here we reduce training ROIs per 
    # image because the images are small and have few objects.
    ROI_POSITIVE_RATIO = 0.25
    TRAIN_ROIS_PER_IMAGE = 128

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (128, 128)  # (height, width) of the mini-mask

    # Maximum number of ground truth instances in one image
    MAX_GT_INSTANCES = 120

    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 100

    # Image mean pixel value per channel (RGB)
    # Values computed for the solar dataset in inspect_solar_data.ipynb
    MEAN_PIXEL = np.array([112.047, 117.697, 107.700])

    # Number of color channels per image.
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 3


################################################################
#  Override Mask R-CNN Dataset class
################################################################

class SolarDataset(utils.Dataset):

    def load_solar(self, dataset_dir, subset=None, resize_dataset=None, 
                   train_test_split=None, is_train=False, is_val=False, is_test=False):
        """
        Load the Solar dataset.
            dataset_dir: root directory of the dataset
            subset: if separated in different folders, subset to load ('train' or 'val')
            resize_dataset: nb of images to keep from a large dataset
            train_test_split: proportion of data to keep for training (the remaining data 
                              is split between validation and test set)
            is_train: boolean, if loading training dataset from train_test_split
            is_val: boolean, if loading validation dataset from train_test_split
            is_test: boolean, if loading test dataset from train_test_split
        """
        # Add classes - here, we have only one class to add.
        # We pass the dataset name, nb of classes and class name.
        self.add_class("solar", 1, "solar array")

        # Select between train and validation dataset if relevant
        if subset in ["train", "val"]:
            dataset_dir = os.path.join(dataset_dir, subset)

        # Resize dataset if requested
        image_ids = os.listdir(dataset_dir)
        size_dataset = len(image_ids)
        if resize_dataset != None:
            size_dataset = resize_dataset
            image_ids = random.sample(image_ids, size_dataset)

        # Train-test split if requested
        if train_test_split != None:
            nb_train = int(train_test_split * size_dataset) # set the nb where to cut
            nb_val = int((train_test_split + (1-train_test_split)/2) * size_dataset)
            random.Random(5).shuffle(image_ids) # randomly shuffle the dataset (with seed)
            if is_train:
                image_ids = image_ids[:nb_train]
                print('Split ratio = {}%'.format(int(train_test_split * 100)))
                print('Nb of training images = {}'.format(len(image_ids)))
            if is_val:
                image_ids = image_ids[nb_train:nb_val]
                print('Split ratio = {}%'.format(int(((1 - train_test_split)/2) * 100)))
                print('Nb of validation images = {}'.format(len(image_ids)))
            if is_test:
                image_ids = image_ids[nb_val:]
                print('Split ratio = {}%'.format(int(((1 - train_test_split)/2) * 100)))
                print('Nb of test images = {}'.format(len(image_ids)))

        # Set up the list of images corresponding to the dataset we load
        for image_id in image_ids:
            self.add_image(
                "solar",
                image_id=image_id[:-4],
                path=os.path.join(dataset_dir, image_id))

    def load_mask(self, image_id, masks_dir=None):
        """
        Load instance masks for an image.
            image_id: the image_id as defined in load_solar
            masks_dir: necessary if masks aren't a subset of the root image directory
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                   one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        if masks_dir == None:
            mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])),
                                                                "masks")
        else:
        	mask_dir = masks_dir

        # If not a solar dataset image, delegate to parent class
        if info["source"] != "solar":
            return super(self.__class__, self).load_mask(image_id)

        # Read mask files from .png images
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.startswith(info["id"]):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] numpy array (RGB).
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale, convert to RGB for consistency
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "solar":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


################################################################
#  Override Mask R-CNN Inference & Visualization functions
################################################################

class InferenceConfig(SolarConfig):
    # Set batch size to 1 since we'll be running inference on one
    # image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, plot=True, save=None):
    """
    Overwrite the fuction from visualize.py in order to add plot and save
    arguments, which allow to save the results of the model instead of 
    plotting them (used in the flask app).
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = matplotlib.patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=14, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = skimage.measure.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = matplotlib.patches.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if plot:
        if auto_show:
            plt.show()
    if not plot:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0)
        plt.close()


################################################################
#  Add solar utils functions
################################################################

def detect_multi(model, dataset_dir, subset=None, 
                    results_dir=os.path.join(ROOT_DIR,"inference")):
    """
    Run detection on images in a given directory.
    Returns: a file (format TBD) with all polygon coordinates,
             or other relevant results.
    """
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load data to process
    dataset = SolarDataset()
    dataset.load_solar(dataset_dir, subset)
    dataset.prepare()

    # Perform detection over images
    results = []
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        r = model.detect([image], verbose=0)[0]
        # Encode relevant information (TBD)
        info = "TBD"
        results.append(info)
        # Save image with masks (TBD)
        visualize.display_instances(image, r['rois'], r['masks'],
                    r['class_ids'], dataset.class_names, r['scores'], 
                    show_bbox=False, show_mask=False, title="Predictions")
        plt.savefig("{}/{}.png".format(results_dir,
                                       dataset.image_info[image_id]["id"]))
    
    # Save results to csv (or other format, TBD)
    file_path = os.path.join(results_dir, "results.csv")
    with open(file_path, "w") as f:
        f.write(results)
    print("Results saved to ", results_dir)

def compute_mask_to_surface(masks, pixel_to_surface_ratio=(1.75/20)**2):
    """
    Convert pixel area to surface area.
    Default ratio corresponds to the parking detection model.
    """
    surface = np.zeros(masks.shape[2])
    for i in range(0, masks.shape[2]):
        surface[i] = masks[:,:,i].sum() * pixel_to_surface_ratio
    return surface

def color_splash(image, mask):
    """
    Apply color splash effect to an image based on mask.
        image: RGB image as np array [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """

    # Make a grayscale copy of the image - but with still 3 channels.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    
    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    """
    Detect objects in mask or video, applies color_splash function
    and writes a new file.
    """
    assert image_path or video_path

    # Process image
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    
    # Process video
    elif video_path:
        import cv2
        
        # Define video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        # Write all frames of new video with color splash effect
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Apply color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    
    print("New file saved as ", file_name)

def get_area_coordinates(center_lat, center_lng, lateral_size):
    """
    Converts cartesian meters to espg:4326 decimal degrees (approximation
    works for the latitudes close to France) in order to compute coordinates 
    of all small squares (200m lateral size) that can fit within a larger square.
    The obtained coordinates can then be used for Google Solar API queries.
    
        center_lat, center_lng: coordinates of the center of the larger square
        lateral_size: lateral size of the larger square (in cartesian meters)
        
    Returns: a list of (lat, lng) tuples corresponding to coordinates of each
             small square center (in espg:4326).
    """
    n_images = lateral_size // 200
    R = 6378137 # Earth's radius
    coordinates = []
    for x in range(-(n_images - 1) * 100, (n_images) * 100, 200):
        for y in range(-(n_images - 1) * 100, (n_images) * 100, 200):
            # Coordinate offsets in radians
            dLat = y / R
            dLng = x / (R * math.cos(math.pi * center_lat / 180))
            # New coordinates in decimal degrees
            new_lat = center_lat + dLat * 180 / math.pi
            new_lng = center_lng + dLng * 180 / math.pi
            coordinates.append((new_lat, new_lng))
    return coordinates

def mrcnn_masks_to_polygons(masks, pixel_transform,
                            image_crs, dest_crs='epsg:4326'):
    """
    Function for the flask app.
    Use imantics, shapely and pyproj libraries to i) create
    Polygons from MRCNN outputs, ii) transform them from pixel
    space to the source image crs, and iii) finally reproject them to a
    chosen dest crs (which enables plotting, with folium for example).

        masks: MRCNN ['masks'] output
        pixel_transform: affine transformation of the original image
                         (can be obtained with rasterio image.transform)
        image_crs: crs of the original image
        dest_crs: destination crs, by default OSM's crs (4326).

    Returns: a list of shapely polygons in destination crs.
    """
    mask = masks.astype(int)
    mask = mask.sum(axis=2)
    polygons_px = imantics.Polygons.from_mask(mask)
    polygons_dest_crs = []
    for i in range(masks.shape[2]):
        if len(polygons_px.points[i])<3:
            continue
        else:
            polygon = shapely.geometry.Polygon([pixel_transform * point \
                                            for point in polygons_px.points[i]])
            reproject = pyproj.Transformer.from_proj(pyproj.Proj(init=image_crs),
                                                     pyproj.Proj(init='epsg:4326'))
            polygon = shapely.ops.transform(reproject.transform, polygon)
            polygons_dest_crs.append(polygon)
    return polygons_dest_crs

def call_google_earthenginesolar_API(
    LAT, LNG, ID, RADIUS, API_KEY, output_path="./",
    compute_RGB_IMG=True,
    compute_Flux_IMG=True,
    compute_Elevation_IMG=True):
    """Calls GEE static API to download images in the specified folder

       Warnings: Not all areas worlwide are covered. Max radius is 100m.
       API key isn't to be shared without caution.

       Images crs is the local flavour of WSG84 (ex: espg 32631).
       Resolution is 0.1m (i.e. each pixel represents 10cm, 
       which corresponds to 2000 px width for an image with 100m radius).
    """
    urlEarthEngineSolar = (('https://earthenginesolar.googleapis.com/v1/solarInfo:'+
                            'get?location.latitude={lat}&location.longitude={lng}&'+
                            'radiusMeters={radiusM}&view=FULL&key={key}')
                            .format(lat=LAT,lng=LNG, radiusM=RADIUS, key=API_KEY))
    try: 
        urllib.request.urlopen(urlEarthEngineSolar)
    except urllib.error.URLError as error:
        print(e.code,e.reason)
        raise ValueError
    else:
        res = urllib.request.urlopen(urlEarthEngineSolar)
        res_body = res.read()
        sunroofResponseRaster = json.loads(res_body.decode("utf-8"))
        rgbImg = sunroofResponseRaster['rgbUrl']
        fluxImg = sunroofResponseRaster['annualFluxUrl']
        dsmImg = sunroofResponseRaster['dsmUrl']
        if not output_path[-1] is "/":
            output_path += "/"
        if compute_RGB_IMG:
            urllib.request.urlretrieve(rgbImg, '{}_RGB.tif'.format(output_path+str(ID)))
        if compute_Flux_IMG:
            urllib.request.urlretrieve(fluxImg, '{}_FLUX.tif'.format(output_path+str(ID)))
        if compute_Elevation_IMG:
            urllib.request.urlretrieve(dsmImg, '{}_DSM.tif'.format(output_path+str(ID)))

def check_around(lat, lng, radius, tag):
    """
    Use a funciton from osmnx library to check for the presence of 
    certain OSM tags around a lat/lng point.
    Much more options are possible with osmnx - see documentation.
        lat, lng: point coordinates (espg:4326)
        radius: radius of square area to search (in m)
        tag: a valid OSM tag (ex: 'buildings')
    Returns: boolean.
    """
    return len(ox.footprints.footprints_from_point(point=(lat, lng), 
                                       distance=radius, 
                                       footprint_type=tag, 
                                       retain_invalid=False, 
                                       custom_settings=None)) > 0


################################################################
#  Set Command Line interface
################################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect solar arrays.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/solar/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=MODELS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SolarConfig()
    else:
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        #train(model)
        train(model, args.dataset, args.subset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect' or 'splash'".format(args.command))

