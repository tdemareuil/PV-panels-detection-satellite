import os
import sys
import numpy as np
import skimage


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
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_WEIGHTS_PATH):
    utils.download_trained_weights(COCO_WEIGHTS_PATH)

# Path to dataset - change if needed
SOLAR_DIR = os.path.join(ROOT_DIR, "datasets/solar")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)

# Path to results directory (TBD)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


################################################################
#  Configuration
################################################################

class SolarConfig(Config):
    """Configuration for training on the solar dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "solar"

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # We use a GPU with 16GB memory, which can fit several images.
    # Adjust down if you use a smaller GPU.
    # Batch size is (GPUs * images/GPU).
    IMAGES_PER_GPU = 6

    # Backbone network architecture
    # Supported values are: resnet50, resnet101 (default)
    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + solar array

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # You can also choose to not exclude based on confidence. Since we have 
    # 2 classes, 0.5 is the minimum anyway as it picks between solar and BG.
    # DETECTION_MIN_CONFIDENCE = 0

    # Input image resizing. Image size must be dividable by 2 at least 6 times 
    # to avoid fractions when downscaling and upscaling.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_SHAPE = [512, 512, 3]
    # IMAGE_MIN_SCALE = 2.0

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # square anchor side in pixels

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

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
    MINI_MASK_SHAPE = (256, 256)  # (height, width) of the mini-mask

    # Maximum number of ground truth instances to use in one image
    # MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 400


################################################################
#  Dataset
################################################################

class SolarDataset(utils.Dataset):

    def load_solar(self, dataset_dir, subset=None, resize_dataset=None, 
                   train_test_split=None, is_train=False, is_val=False):
        """
        Load a subset of the Solar dataset.
            dataset_dir: Root directory of the dataset.
            subset: Subset to load: train or val, if separated in different folders
            resize_dataset: nb of images to keep from large dataset
            train_test_split: if desired, proportion of data to keep for training vs. val
            is_train: boolean, if loading training data
            is_val: boolean, if loading validation data
        """
        # Add classes. We have only one class to add.
        # We pass the dataset name, nb of classes and class name.
        self.add_class("solar", 1, "solar array")

        # Select between train and validation dataset if required
        if subset in ["train", "val"]:
            dataset_dir = os.path.join(dataset_dir, subset)

        # Resize dataset if required
        image_ids = os.listdir(dataset_dir)
        size_dataset = len(image_ids)
        if resize_dataset != None:
            size_dataset = resize_dataset
            image_ids = random.sample(image_ids, size_dataset)
        # print("Nb of images in dataset = ", size_dataset)

        # Train-test split if required
        if train_test_split != None:
            nb_train = int(train_test_split * size_dataset)
            if is_train:
                image_ids = image_ids[:nb_train]
                print('Split ratio = {}%'.format(int(train_test_split * 100)))
                print('Nb of training images = {}'.format(len(image_ids)))
            if is_val:
                image_ids = image_ids[nb_train:]
                print('Split ratio = {}%'.format(int(train_test_split * 100)))
                print('Nb of validation images = {}'.format(len(image_ids)))

        # Get clean image ids for loading
        image_ids = [image_id[:-4] for image_id in image_ids]

        # Add images
        for image_id in image_ids:
            self.add_image(
                "solar",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id + ".png"))

    def load_mask(self, image_id, masks_dir=None):
        """
        Load instance masks for an image.
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

        # If not a solar dataset image, delegate to parent class.
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

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "solar":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def add_rel_lum(image_id):
        """Add relative luminance (Y channel) to the image
           with formula: Y = 0.2126R + 0.7152G + 0.0722B
           Args: image id or image as np array? (TBD)
        """
        image = skimage.io.imread(image_id)
        rel_lum_layer = 0.2126*image[:,:,0] + 0.7152*image[:,:,1] + 0.0722*image[:,:,2]
        new_image = np.dstack((image, rel_lum_layer))
        return new_image


################################################################
#  Detection
################################################################

class InferenceConfig(SolarConfig):
    # Set batch size to 1 since we'll be running inference on one
    # image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False, masks_dir=None):
    """Overwrites the base function from model.py, in order to add an argument
       to specify the masks folder.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    if masks_dir == None:
    	mask, class_ids = dataset.load_mask(image_id)
    else:
    	mask, class_ids = dataset.load_mask(image_id, masks_dir=masks_dir)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = modellib.compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask

def detect_multi(model, dataset_dir, subset=None, results_dir=RESULTS_DIR):
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


################################################################
#  Command Line
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
                        default=DEFAULT_LOGS_DIR,
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

