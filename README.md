# Solar Panels detection in satellite imagery
Thomas de Mareuil - Total E-LAB - May 2020

* Repository

This repository holds the files for a satellite image analysis app able to detect solar panels from Google high-resolution overhead imagery.

The repository is organized as follows:

```bash
.
├── data
│   ├── images [1906 entries]
│   └── masks [5219 entries]
├── inference
│   └── downloaded images and detection outputs
├── models
│   └── trained weights
├── notebooks
│   └── notebooks to inspect data and model
├── templates
│   └── HTML templates for the app
├── static
│   └── images and gifs served in the app
│
├── __init__.py
├── documentation.md
├── requirements.txt
├── solar_app.py
└── solar.py
```


* Model

The model we use is **Mask R-CNN** (see original [paper](https://arxiv.org/abs/1703.06870) by FAIR, 2017, and Python [implementation](https://github.com/matterport/Mask_RCNN) by Matterport, Inc, which we adapted to our use case). The `mrcnn` library is located in the root directory of the Demeter project, and the `solar.py` file in this folder holds all the functions specific to our solar panels segmentation task (i.e. functions overwriting `mrcnn` classes and new solar-specific utils functions).

Mask R-CNN is considered as state-of-the-art (as of 2020) to perform **instance segmentation** tasks. It is based on a CNN backbone (here, ResNet101) with a Feature Pyramid Network (FPN) architecture, next to a Region Proposal Network (RPN - similar to the Faster R-CNN model), which are finally followed by customized head branches for bounding box regression and pixel-level (mask) classification. See below 2 diagrams representing the model architecture:

![Mask R-CNN architecture 1](https://miro.medium.com/max/3240/1*M_ZhHp8OXzWxEsfWu2e5EA.png)

![Mask R-CNN architecture 2](https://www.researchgate.net/profile/Lukasz_Bienias/publication/337795870/figure/fig2/AS:834563236429826@1575986789511/The-structure-of-the-Mask-R-CNN-architecture.png)

More details on Mask R-CNN can be found in this [Medium article](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272) and this [presentation](https://www.slideshare.net/windmdk/mask-rcnn) by the authors of the original paper at ICCV 2017.


* Training

The data we used for training is made available by Duke University at this [address](https://figshare.com/collections/Full_Collection_Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3255643). It includes high resolution satellite imagery (0.3m, 2016) for 4 cities in California, with polygon annotations corresponding to solar arrays. We pre-processed the data to obtain image tiles of 500px width (1906 images) and their corresponding binary masks (5219 masks, i.e. solar array instances). You can find extensive training data analysis and multiple visualizations in the `inspect_solar_data.ipynb` notebook.

We trained the model for 100 epochs using a single GPU with 16Gb RAM, and keeping aside 20% of the data for validation. As part of training, we also performed image augmentation (using `imgaug`) and added a 4th channel to each image, corresponding to Relative Luminance (formula: $Y=0.2126R+0.7152G+0.0722B$). We trained both the CNN backbone and head layers, starting from randomly initialized weights.


* Testing

After fine-tuning the data pre-processing, CNN backbone, RPN anchors and thresholds and final detection heads, our model reaches a Mean Average Precision score (mAP) of 0.81 on the validation set.

Mean average precision is a commonly used metric for instance segmentation tasks. It is based on the area under the Precision-Recall curve, for which True Positive detections correspond to predicted masks whose IoU score (Intersection over Union) stands above a certain threshold (for example 0.5). We go through detailed model testing steps in the `inspect_solar_model.ipynb` notebook. See below some detection examples:

Detection example.

You can also find more details on the mAP metric in these 2 blog posts: [link](https://www.jeremyjordan.me/evaluating-image-segmentation-models/) and [link](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52).


* Web application

We built an application with `flask` to serve the model with a user-friendly interface. Requirements can be installed through the `requirements.txt` file, and the app can be run in a new virtual environment as follows:
```bash
pip install requirements.txt
python solar_app.py
```

While running, the app displays status information at each step in the terminal. The static files necessary to run the app are stored in the `static` folder and the HTML templates for the home and success pages are in the `templates` folder. The app uses AJAX, JQuery and Jinja2 frameworks.

In the app, the user can select coordinates (latitude/longitude) and area size from an embedded javascript `Leaflet` map. The app then checks if there are buildings (and therefore, potentially solar panels) in the area through OpenStreetMap's `Overpass` API, and downloads high-resolution satellite images from Google Static API (0.1m resolution, 200m width image tiles). It performs solar panels instance segementation using our trained Mask R-CNN model and redirects to the `\success` (or `\success\batch`) page, where the user can visualize results overlayed on a Leaflet map. They can also download a detection summary (Excel format) and a GeoJSON file holding coordinates for all the polygons corresponding to the detected solar arrays.


* Next steps

This app is to be integrated in Total's Azure ecosystem by the E-LAB back-end developers team, so that several Group subsidiaries (Quadran, Direct Energie, Total Energy Trading) can use it.

In the future, the model, notebooks and application could be adapted by Total E-LAB to other satellite image analysis use cases. We could also imagine a single app serving several object detection models, to perform different tasks from a single interface.
