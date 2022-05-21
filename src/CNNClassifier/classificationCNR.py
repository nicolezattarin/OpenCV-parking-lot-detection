import numpy as np
import pandas as pd
import tensorflow as tf
import os

# import labels 
def get_labels (camera_number, weather):
    """
    returns a dataframe with pairs (filename, labels) for the given camera and weather
    """
    path ="../../CNR-EXT-Patches-150x150/LABELS/camera{}.txt".format(camera_number)
    weater_label = weather[0].upper()
    # print("weater_label ", weater_label)

    # read as a file of strings
    with open(path, 'r') as f:
        data = list(f.read().splitlines())
    
    # get labels 
    data = [[d.split(' ')[0].split("/")[-1], int(d.split(' ')[1])]  for d in data]
    # filter by weather
    data = [d for d in data if d[0][0] == weater_label]
    data = pd.DataFrame(data, columns=['filename', 'label'])
    data['label_name'] = data['label'].apply(lambda x: 'busy' if x == 1 else 'free')

    return data

def classify (model, path):
    """
    classify image at a given path
    params:
        model: trained model
        path: path to image
    return:
        class_id: class id of the image
        class_name: class name of the image
        confidence: confidence of the model
    """

    img = tf.keras.utils.load_img(path, target_size=(150, 150))    
    img_array = tf.keras.utils.img_to_array(img)
    #normalize array
    img_array = img_array / 255.
    img_array = tf.expand_dims(img_array, 0) 
    
    predictions = model.predict(img_array, 1)
    class_id = 0 if predictions[0][0] > 0.5 else 1
    class_name = 'busy' if class_id == 1 else 'free'
    confidence = predictions[0][0] if class_id == 0 else 1-predictions[0][0]
    return class_id, class_name, confidence*100
    

def load_model(batch_size, epochs):
    model_dir = "CNN_model/{}_epochs_{}_batch_classifier".format(epochs, batch_size)
    model = tf.keras.models.load_model(model_dir)
    return model

def classify_data (camera_number, weather, batch_size, epochs, n_imgs=None):
    """
    classify all images in the given camera and weather
    params:
        camera_number: camera number
        weather: weather
    return:
        dataframe with pairs (filename, class_id, class_name, confidence)
    """
    # example of path
    # PATCHES_PROCESSED/SUNNY/camera1/S_2015-11-12_07.09_C01_184.jpg

    # dir = "../../PATCHES_PROCESSED/"+weather.upper()+"/camera"+str(camera_number)
    dir ='../../CNR-EXT-Patches-150x150/PATCHES/SUNNY/2015-11-12/camera1'
    import os
    if not os.path.exists(dir): raise Exception("path does not exist")
        
    import glob
    filenames = glob.glob(dir+"/*.jpg")
    if len(filenames) == 0: raise Exception("images not found")

    if n_imgs is not None:
        if len(filenames) < n_imgs: raise Exception("not enough images")
        else: filenames = filenames[:n_imgs]

    model = load_model(batch_size=batch_size, epochs=epochs)
    data = []
    i = 0
    for filename in filenames:
        if i % 50 == 0: print("classifying {} of {}".format(i, len(filenames)))
        class_id, class_name, confidence = classify(model, filename)
        f = filename.split("/")[-1]
        data.append([f, class_id, class_name, confidence])
        i+=1
    data = pd.DataFrame(data, columns=['filename', 'class_id', 'class_name', 'confidence'])
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
parser.add_argument("camera_number", type=int)
parser.add_argument("weather", type=str)
parser.add_argument("--nimgs", type=int, default=200)
parser.add_argument("--preprocessing", type=str, default="none")

def main (camera_number, weather, nimgs, preprocessing):
    classified_sample = classify_data (1, 'sunny', batch_size=32, epochs=10, n_imgs=nimgs)
    labels = get_labels(1, 'sunny')
    accuracy = get_accuracy(classified_sample, labels)
    
    #save results 
    if not os.path.exists("../../results/CNR/camera{}/{}".format(camera_number, weather)):
        os.makedirs("../../results/CNR/camera{}/{}".format(camera_number, weather))
    classified_sample.to_csv("../../results/CNR/camera{}/{}/classified_sample_preproc_{}.csv".format(camera_number, weather, preprocessing))
    file = open("../../results/CNR/camera{}/{}/accuracy_preproc_{}.txt".format(camera_number, weather, preprocessing), "w")
    file.write("accuracy: {}".format(accuracy))
    file.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))