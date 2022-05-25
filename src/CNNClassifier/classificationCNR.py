import numpy as np
import pandas as pd
import tensorflow as tf
import os

# import labels 
def get_labels (camera_number, weather):
    """
    returns a dataframe with pairs (filename, labels) for the given camera and weather
    params:
        camera_number: camera number
        weather: weather
    return:
        dataframe with  (filename, labels)
    """
    path ="../../CNR-EXT-Patches-150x150/LABELS/camera{}.txt".format(camera_number)
    weather_label = weather[0].upper()
    # print("weather_label ", weater_label)

    # read as a file of strings
    with open(path, 'r') as f:
        data = list(f.read().splitlines())
    
    # get labels 
    data = [[d.split(' ')[0].split("/")[-1], int(d.split(' ')[1])]  for d in data]
    # filter by weather
    data = [d for d in data if d[0][0] == weather_label]
    data = pd.DataFrame(data, columns=['filename', 'label'])
    data['label_name'] = data['label'].apply(lambda x: 'busy' if x == 1 else 'free')

    return data

def get_accuracy(data, labels):
    """
    returns accuracy of the model
    params:
        data: dataframe with pairs (filename, class_id, class_name, confidence)
        labels: dataframe with pairs (filename, label)
    return:
        accuracy
    """
    data = data.merge(labels, on='filename')
    data['correct'] = data['class_id'] == data['label']
    return data['correct'].mean()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--camera_number", type=int, default=1)
parser.add_argument("--weather", type=str, default='sunny')
parser.add_argument("--nimgs", type=int, default=None)
parser.add_argument("--preprocessing", type=str, default="none")
parser.add_argument("--dir_index", type=str, default="000") 
                                                            

def main (camera_number, weather, nimgs, preprocessing, dir_index):
    """
    classifies the images in the given camera and weather
    params:
        camera_number: camera number
        weather: weather
        nimgs: number of images to classify
        preprocessing: preprocessing
        dir_index: directory index, eg camera_000, camera_001 according to the kind of preprocess, 
                    should be made automatic in a second version of the code order
                    is rot-eq-blur as in preprocessing/main.cpp

    """
    from utils import classify_data

    classified_sample = classify_data (camera_number, weather, dir_index, batch_size=32, epochs=10, n_imgs=nimgs, dataset="CNR")
    labels = get_labels(camera_number, weather)
    accuracy = get_accuracy(classified_sample, labels)
    
    #save results 
    if not os.path.exists("../../results/CNR/camera{}/{}".format(camera_number, weather)):
        os.makedirs("../../results/CNR/camera{}/{}".format(camera_number, weather))
    classified_sample.to_csv("../../results/CNR/camera{}/{}/classified_sample_preproc_{}.csv".format(camera_number, weather, preprocessing), index=False)
    file = open("../../results/CNR/camera{}/{}/accuracy_preproc_{}.txt".format(camera_number, weather, preprocessing), "w")
    file.write("accuracy: {}".format(accuracy))
    file.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))