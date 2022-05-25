from warnings import WarningMessage
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import argparse
import tensorboard as tb
import pandas as pd
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

###########################################################################
#                                 TRAINING                                #
###########################################################################

def prepare_data(busy_data_path, free_data_path, val_split=0.25, train_split=0.5, test_split=0.25, save=False, save_file='classifier_data', nmax=None):
    """
    Prepare data for training and testing. this function could need a lot of time and memory, 
    so it is recommended to run it in a separate process.
    Moreover, the option save=True will save the data in such a way to make it possible
    to process data with tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory

    params:
        busy_data_path: path to busy data
        free_data_path: path to free data
        val_split: percentage of data to be used for validation
        train_split: percentage of data to be used for training
        test_split: percentage of data to be used for testing
        save: if True, the data will be saved in such a way to make it possible to use a generator
        save_file: name of the file to save the data
        nmax: maximum number of images to use

    returns:
        train_data, val_data, test_data: prepared data
    """
    import glob
    import cv2

    if val_split + train_split + test_split != 1:
        raise ValueError("val_split + train_split + test_split must equal 1")
    print('Loading data...')
    if nmax is not None:
        images_busy = [cv2.imread(file) for file in glob.glob(busy_data_path)[:nmax]]
        images_free = [cv2.imread(file) for file in glob.glob(free_data_path)[:nmax]]
    else:
        images_busy = [cv2.imread(file) for file in glob.glob(busy_data_path)]
        images_free = [cv2.imread(file) for file in glob.glob(free_data_path)]

    #shuffle the data
    np.random.shuffle(images_busy)
    np.random.shuffle(images_free)

    #split
    train_busy = images_busy[:int(len(images_busy)*train_split)]
    val_busy = images_busy[int(len(images_busy)*train_split):int(len(images_busy)*(train_split+val_split))]
    test_busy = images_busy[int(len(images_busy)*(train_split+val_split)):]

    train_free = images_free[:int(len(images_free)*train_split)]
    val_free = images_free[int(len(images_free)*train_split):int(len(images_free)*(train_split+val_split))]
    test_free = images_free[int(len(images_free)*(train_split+val_split)):]
    print("train_busy:", len(train_busy), "val_busy:", len(val_busy), "test_busy:", len(test_busy))
    #save the data if you want to use it later
    if save:
        import os 
        if not os.path.isdir("../"+save_file):
            os.mkdir("../"+save_file)

        if not os.path.isdir("../"+save_file+"/train"):
            os.mkdir("../"+save_file+"/train")
            os.mkdir("../"+save_file+"/train/busy")
            os.mkdir("../"+save_file+"/train/free")

            print("saving train imgs")
            for i in range(len(train_free)):
                cv2.imwrite("../"+save_file+"/train/free/"+str(i)+"_train_free.jpg", train_free[i])
            for i in range(len(train_busy)):
                cv2.imwrite("../"+save_file+"/train/busy/"+str(i)+"_train_busy.jpg", train_busy[i])

        if not os.path.isdir("../"+save_file+"/test"):
            os.mkdir("../"+save_file+"/test")
            os.mkdir("../"+save_file+"/test/busy")
            os.mkdir("../"+save_file+"/test/free")
            print("saving test imgs")
            for i in range(len(test_free)):
                cv2.imwrite("../"+save_file+"/test/free/"+str(i)+"_test_free.jpg", test_free[i])
            for i in range(len(test_busy)):
                cv2.imwrite("../"+save_file+"/test/busy/"+str(i)+"_test_busy.jpg", test_busy[i])

        if not os.path.isdir("../"+save_file+"/val"):
            print("saving validation imgs")
            os.mkdir("../"+save_file+"/val")
            os.mkdir("../"+save_file+"/val/busy")
            os.mkdir("../"+save_file+"/val/free")
            for i in range(len(val_free)):
                cv2.imwrite("../"+save_file+"/val/free/"+str(i)+"_val_free.jpg", val_free[i])
            for i in range(len(val_busy)):
                cv2.imwrite("../"+save_file+"/val/busy/"+str(i)+"_val_busy.jpg", val_busy[i])
    
    return train_busy, train_free, val_busy, val_free, test_busy, test_free


def frozen_graph_maker(model, output_graph_path):
    """
    NOTE: REQUIRES NO MORE THAN TENSORFLOW 2.2
    convert saved_model.pb to frozen graph taken from:
    https://medium.com/@sebastingarcaacosta/how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb
    we need to save a frozen graph to upload the model in opencv c++, otherwise the error
    "'opencv_tensorflow.FunctionDef.Node.ret' contains invalid UTF-8 data when parsing a protocol buffer." occurs
    """
    #this solution runs only on tensorflow 2.2
    if not tf.__version__.startswith('2.2'):
        raise WarningMessage("This function requires tensorflow 2.2")
        return 
    # convert the model to a graph
    from tensorflow.python.framework.convert_to_constant import convert_variables_to_constants_v2

    full_model = tf.function(lambda x:model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    
    #save
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=output_graph_path, name="frozen_graph.pb", as_text=False)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=output_graph_path, name="frozen_graph.pbtxt", as_text=True)
    

def get_test_generator(test_dir, batch_size):
    """
    returns a generator for the test data
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(directory=test_dir, batch_size=batch_size,
                                                        class_mode='binary', target_size=(150,150))
    return train_generator


###########################################################################
#                             CLASSIFICATION                              #
###########################################################################

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
    

def load_model(batch_size, epochs, dir_name):
    """
    Loads a model from a given directory
    params:
        batch_size: batch size of the model
        epochs: epochs of the model
        dir_name: directory name of the model
    return:
        model: loaded model
    """
    model_dir = dir_name+"/{}_epochs_{}_batch_classifier".format(epochs, batch_size)
    model = tf.keras.models.load_model(model_dir)
    return model

def classify_data (camera_number, weather, dir_index, dataset, batch_size=32, epochs=10, n_imgs=None):
    """
    classify all images in the given camera and weather
    params:
        camera_number: camera number
        weather: weather
        dir_index: directory index
        dataset: dataset
        batch_size: batch size of the model
        epochs: epochs of the model
        n_imgs: number of images to classify
    return:
        dataframe with  (filename, class_id, class_name, confidence)
    """
    # example of path
    # PATCHES_PROCESSED/SUNNY/camera1/S_2015-11-12_07.09_C01_184.jpg

    dir = "../../"+dataset.upper()+"_PATCHES_PROCESSED/"+weather.upper()+"/camera"+str(camera_number)+"_"+dir_index
    import os
    if not os.path.exists(dir): raise Exception("path does not exist")
        
    import glob
    filenames = glob.glob(dir+"/*.jpg")
    if len(filenames) == 0: raise Exception("images not found")

    if n_imgs is not None:
        if len(filenames) < n_imgs: raise Exception("not enough images")
        else: filenames = filenames[:n_imgs]

    model = load_model(batch_size=batch_size, epochs=epochs, dir_name="CNN_model"+dataset.upper())
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

