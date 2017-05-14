# SynDataGeneration 

This code is used to generate synthetic scenes for the task of instance/object detection. Given images of objects in isolation from multiple views and some background scenes, it generates full scenes with multiple objects and annotations files which can be used to train an object detector. The approach used for generation works welll with region based object detection methods like [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn).

## Pre-requisites 
1. OpenCV (pip install opencv-python)
2. PIL (pip install Pillow)
3. Poisson Blending (Follow instructions [here](https://github.com/yskmt/pb)
4. PyBlur (pip install pyblur)

To be able to generate scenes this code assumes you have the object masks for all images. There is no pre-requisite on what algorithm is used to generate these masks as for different applications different algorithms might end up doing a good job. However, we recommend Graph Cut or Pixel Objectness with CRF/Bilinear Pooling to generate these masks.

## Setting up Defaults

## Running the Script

## Training an object detector
The code produces all the files required to train an object detector. The format is directly useful for Faster R-CNN but might be adapted for different object detectors too. The different files produced are:
1. __labels.txt__ - Contains the labels of the objects being trained
2. __annotations/*.xml__ - Contains annotation files in XML format which contain bounding box annotations for various scenes
3. __images/*.jpg__ - Contain image files of the synthetic scenes in JPEG format 
4. __train.txt__ - Contains list of synthetic image files and corresponding annotation files

There are tutorials describing how one can adapt Faster R-CNN code to run on a custom dataset like:
1. https://github.com/rbgirshick/py-faster-rcnn/issues/243
2. http://sgsai.blogspot.com/2016/02/training-faster-r-cnn-on-custom-dataset.html
