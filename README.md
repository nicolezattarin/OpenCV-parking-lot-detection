# OpenCV parking lot detection
 
We provide a possible strategy for detecting parking lots in a given camera image, the design is based on [OpenCV](https://opencv.org/) and a CNN classifier implementedn using [TensorFlow](https://www.tensorflow.org/).

The code i adapted to analyze images coming from [CNRPark+EXT](http://cnrpark.it) dataset, and [Parking Lot database](https://web.inf.ufpr.br/vri/databases/parking-lot-database/).

## Typical workflow
The src folder contains the code for the classifier, pre- and post-processing of images,data are meant to be saved in the parent directory.
A typical workflow follows.

### CNR DATASET
To download the datasets run:
```
git clone https://github.com/nicolezattarin/OpenCV-parking-lot-detection.git
cd OpenCV-parking-lot-detection
wget http://cnrpark.it/dataset/CNR-EXT_FULL_IMAGE_1000x750.tar
tar -xvf CNR-EXT_FULL_IMAGE_1000x750.tar
wget http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip
unzip CNR-EXT-Patches-150x150.zip
wget http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip
unzip CNRPark-Patches-150x150.zip

````

Then, manually check and put the right path in utils.cpp (lines 145, 158), and perform preprocessing:

```
cd src/preprocessing
# compile and link
make
# run the preprocessing, e.g. ./main 1 sunny 0 1 0 cnr
./main <camera_number> <weather> <flag_rotation> 
	 <flag_equalization> <flag_blur> <dataset> <nimgs>
```
Once the images are ready, you need to eventually train the classifier:
```
cd src/CNNClassifier
python trainingCNR.py --epochs <epochs> --batch_size <batch_size> --load False --save True
```
make sure that the first time you run the script the flag save is True, this allows to prepare data in the correct way for the generator to work, after the first time it is recommendend to set it to False for time complexity reasons.

Then, you can perform the actual detection:
```
python classificationCNR.py --camera_number <camera_number> --weather <weather> 
			    --rot  <flag_rotation> --eq  <flag_equalization> --blur <flag_blur>
```
with the parameters that can be retrieved by simply running `python classificationCNR.py --h`.

Finally, you can have a direct access to results with a final post-processing:
```
cd ../postprocessing
make
./main <camera_number> <weather> <flag_rotation> 
	 <flag_equalization> <flag_blur> <dataset> <nimgs>
```
### PARKING LOT DATABASE

To download the dataset, just run from parent directory:
```
# download data
wget http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
gunzip  PKLot.tar.gz
```
A similar path can be followed to process images from Pklot, but in this case it is recomended to reduce the amount of data to prcess by running first:

# for each camera and weather condition run
reduce_xml_parser.py 
		  --ndata <ndata>
 		  --camera_number <camera_number> 
		  --weather <weather>
``

## Results
Let us first providea proof of the performances of the implemented CNN classifier. The following figure shows the history for train, test and validation accuracy/loss of a model trained with 10 epochs and batch size 32.

<p align="center">
  <img src="figs/10_epochs_32_batch_history.png" width="500" />

</p>

An example of final result, i.e. result after post-processing, is shown in the following figure:

<p align="center">
  <img src="figs/2015-11-12_1239.jpg" width="400" />
</p>

Finally, we also provide the tools to evaluate the performances of our model and test different possible pre-processing configurations.
The following figure shows the distribution of confidence scores for the detected parking lots and the counting of free/busy lots in absence of any pre-processing, which corresponds to an accuracy of 0.85.

<p align="center">
  <img src="figs/confidence_none.png" width="750" />
</p>

***Requirements***

Main requirements are:

- Compiler (g++) compatible with C++17
- Tensorflow >= 2.5
- OpenCV 4.5

Other minor python packages that we use are: pandas, numpy, os, argparse, waring, matplotlib, seaborn, tensorboard, errno, cv2

