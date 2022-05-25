import pandas as pd
import glob
import os

def get_labels (camera_number, weather):
    """
    returns a dataframe with pairs (filename, labels) for the given camera and weather
    params:
        camera_number: camera number
        weather: weather
    """
    paths = "../../PKLot_reduced/camera{}/{}/*_*.csv".format(camera_number, weather.upper())
    filenames = glob.glob(paths)
    data = pd.DataFrame()
    for f in filenames:
        d = pd.read_csv(f)
        d['filename'] = d['filename'].apply(lambda x: x[x.rfind("/")+1:])
        d['date'] = d['filename'].apply(lambda x: x[x.rfind("/")+1:x.rfind("_")-6])
        d['time'] = d['filename'].apply(lambda x: x[x.rfind("_")-5:x.rfind(".")])
        data=pd.concat([data, d])
        
    return data

def get_accuracy(data, labels):
    """
    returns accuracy of the model
    params:
        data: dataframe  
        labels: dataframe 
    return:
        accuracy
    """
    data = data.sort_values(by=['filename'])
    label = labels.sort_values(by=['filename'])
    data['correct'] = data['class_id'] == label['occupied']
    return data['correct'].mean()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--camera_number", type=int, default=3)
parser.add_argument("--weather", type=str, default='rainy')
parser.add_argument("--preprocessing", type=str, default="none")
parser.add_argument("--dir_index", type=str, default="000") # eg camera_000, camera_001 according to the kind of preprocess, 
                                                            # should be made automatic in a second version of the code order 
                                                            # is rot-eq-blur as in preprocessing/main.cpp 

def main (camera_number, weather, preprocessing, dir_index):
    """
    params:
        camera_number: camera number
        weather: weather
        preprocessing: preprocessing
        dir_index: directory index
    """
    from utils import classify_data

    classified_sample = classify_data (camera_number, weather, dir_index, batch_size=32, epochs=10, dataset="PKLOT")
    classified_sample['date'] = classified_sample['filename'].apply(lambda x: x[:x.rfind("_")-6])
    classified_sample['time'] = classified_sample['filename'].apply(lambda x: x[x.rfind("_")-5:x.rfind(".")-4])
    classified_sample['id'] = classified_sample['filename'].apply(lambda x: int(x[x.rfind("_")+4:x.rfind("_")+7]))

    labels = get_labels(camera_number, weather)
    accuracy = get_accuracy(classified_sample, labels)
    
    #save results 
    if not os.path.exists("../../results/Pklot/camera{}/{}".format(camera_number, weather)):
        os.makedirs("../../results/Pklot/camera{}/{}".format(camera_number, weather))
    classified_sample.to_csv("../../results/Pklot/camera{}/{}/classified_sample_preproc_{}.csv".format(camera_number, weather, preprocessing), index=False)
    file = open("../../results/Pklot/camera{}/{}/accuracy_preproc_{}.txt".format(camera_number, weather, preprocessing), "w")
    file.write("accuracy: {}".format(accuracy))
    file.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))