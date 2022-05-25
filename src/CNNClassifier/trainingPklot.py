from warnings import WarningMessage
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import argparse
import tensorboard as tb
import cv2
import errno
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from utils import prepare_data, get_test_generator, frozen_graph_maker


parser = argparse.ArgumentParser(description='Train a classifier')
parser.add_argument('--epochs', type=int , default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int , default=32, help='batch size')
parser.add_argument('--load', type=bool , default=True, help='if true an existing model is loaded')
parser.add_argument('--save', type=bool , default=False, help='if truethe images are saved again from zero')

def main(epochs, batch_size, load, save):
    from CNNClassifier import CNNClassifier
    
    busy_data = "../../PKLot/PKLotSegmented/PUC/*/*/Occupied/*.jpg"
    free_data = "../../PKLot/PKLotSegmented/PUC/*/*/Empty/*.jpg"
    model_dir = "CNN_modelPKlot/{}_epochs_{}_batch_classifier".format(epochs, batch_size)
    
    if load:
        if not os.path.exists(model_dir): 
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_dir) 

        #load the model
        model = tf.keras.models.load_model(model_dir)
        history = np.load("CNN_modelPklot/{}_epochs_{}_batch_history.npy".format(epochs, batch_size), allow_pickle=True).item()

        print("\nmodel loaded")

        #TESTING
        test_gen = get_test_generator("../classifier_dataPklot/test", batch_size)
        print("test data prepared")
        model = tf.keras.models.load_model(model_dir)
        r = model.evaluate(test_gen)
        print("test loss, test acc:", r)
        #save test results
        np.save("CNN_modelPklot/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size), r)

    else:
        print("\npreparing data")
        if save: _, _, _, _, _, _ = prepare_data(busy_data, free_data, save=True, save_file='classifier_dataPklot',nmax= 3000)
        print("data prepared")
        classifier = CNNClassifier(150,150)
        history = classifier.train( train_dir="../classifier_dataPklot/train" , val_dir="../classifier_dataPklot/val",
                                    batch_size=batch_size, epochs=epochs, save_dir="CNN_modelPklot")

        #TESTING
        test_gen = get_test_generator("../classifier_dataPklot/test", batch_size)
        print("test data prepared")
        model = tf.keras.models.load_model(model_dir)
        r = model.evaluate(test_gen)
        print("test loss, test acc:", r)
        #save test results
        np.save("CNN_modelPklot/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size), r)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
