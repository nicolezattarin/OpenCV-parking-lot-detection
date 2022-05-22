# OpenCV parking lot detection
 
We provide a possible strategy for detecting parking lots in a given camera image, the design is based on [OpenCV](https://opencv.org/) and a CNN classifier implementedn using [TensorFlow](https://www.tensorflow.org/).

The code is, at the state of art, adapted to analyze images coming from [CNRPark+EXT](http://cnrpark.it) dataset, even though we plan to extend the study to [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) as well.

### Structure of the code
The src folder contains the code for the classifier, pre- and post-processing of images,data are meant to be saved in the parent directory.
A typical workflow is:
```
# Download the dataset
git clone https://github.com/nicolezattarin/OpenCV-parking-lot-detection.git
cd OpenCV-parking-lot-detection
wget http://cnrpark.it/dataset/CNR-EXT_FULL_IMAGE_1000x750.tar
tar -xvf CNR-EXT_FULL_IMAGE_1000x750.tar
wget http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip
unzip CNR-EXT-Patches-150x150.zip
wget http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip
unzip CNRPark-Patches-150x150.zip

#manually check and put the right path in utils.cpp 
cd src 
cd preprocessing
make

```
