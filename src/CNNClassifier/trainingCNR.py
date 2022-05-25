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

def plot_history(history_trend, epochs, batch_size):
    # PLOT HISTORY OF VALIDATION AND TRAINING ACCURACY/LOSS
    sns.set_theme(style="white", font_scale=1.5, palette="Dark2")

    if epochs == 15: 
        epochs_range = range(1, len(history_trend["accuracy"]))
        history_trend['accuracy'], history_trend['val_accuracy'], history_trend['loss'], history_trend['val_loss'] =\
            history_trend['accuracy'][:-1], history_trend['val_accuracy'][:-1], history_trend['loss'][:-1], history_trend['val_loss'][:-1]
    else: epochs_range = range(1, len(history_trend["accuracy"]) +1 )
    
    #load test results 
    if not os.path.exists("CNN_modelCNR/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size)):
        print("test data not found")
        return
    test_results = np.load("CNN_modelCNR/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size))
    lw=2
    ls='--'
    m = 'o'
    ms = 8
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(epochs_range, history_trend['accuracy'], lw=lw, ls=ls, label='train', ax=ax, marker='o', markersize=ms)
    sns.lineplot(epochs_range, history_trend['val_accuracy'], lw=lw, ls=ls, label='validation', ax=ax, marker='o', markersize=ms)

    # An inner plot to show the peak frequency
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax,  "55%", "60%" ,loc="lower right", borderpad=1)

    sns.lineplot(epochs_range, history_trend['loss'], lw=lw, ls=ls,label='train loss', ax=axins, marker='o', markersize=ms)
    sns.lineplot(epochs_range, history_trend['val_loss'], lw=lw, ls=ls,label='validation loss', ax=axins, marker='o', markersize=ms)

    #dashed lines with accuracy and loss on test set
    ax.axhline(y=test_results[1], label='test', color='black', lw=2)
    axins.axhline(y=test_results[0], label='test', color='black', lw=2)

    ax.legend(loc='lower left')
    axins.get_legend().remove()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    axins.set_ylabel('Loss')
    axins.set_xticklabels([])
    axins.set_yticks([0,0.5,1])
    ax.set_ylim(0,1)
    axins.set_ylim(0,1)
    fig.savefig("CNN_modelCNR/{}_epochs_{}_batch_history.png".format(epochs, batch_size), bbox_inches='tight')


parser = argparse.ArgumentParser(description='Train a classifier')
parser.add_argument('--epochs', type=int , default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int , default=32, help='batch size')
parser.add_argument('--load', type=bool , default=True, help='if true an existing model is loaded')
parser.add_argument('--save', type=bool , default=False, help='if truethe images are saved again from zero')

def main(epochs, batch_size, load, save):
    from CNNClassifier import CNNClassifier
    busy_data = "../../CNRPark-Patches-150x150/*/busy/*.jpg"
    free_data = "../../CNRPark-Patches-150x150/*/free/*.jpg"
    model_dir = "CNN_modelCNR/{}_epochs_{}_batch_classifier".format(epochs, batch_size)
    
    if load:
        if not os.path.exists(model_dir): 
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_dir) 

        #load the model
        model = tf.keras.models.load_model(model_dir)
        history = np.load("CNN_modelCNR/{}_epochs_{}_batch_history.npy".format(epochs, batch_size), allow_pickle=True).item()

        print("\nmodel loaded")

        #TESTING
        test_gen = get_test_generator("../../classifier_dataCNR/test", batch_size)
        print("test data prepared")
        model = tf.keras.models.load_model(model_dir)
        r = model.evaluate(test_gen)
        print("test loss, test acc:", r)
        #save test results
        np.save("CNN_modelCNR/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size), r)
        plot_history(history, epochs, batch_size)

    else:
        if save: _, _, _, _, _, _ = prepare_data(busy_data, free_data, save=True,  save_file='classifier_dataCNR')
        classifier = CNNClassifier(150,150)
        history = classifier.train(batch_size=batch_size, epochs=epochs, save_dir="CNN_modelCNR",
                                    train_dir="../../classifier_dataCNR/train" , val_dir="../../classifier_dataCNR/val")

        #TESTING
        test_gen = get_test_generator("../../classifier_dataCNR/test", batch_size)
        print("test data prepared")
        model = tf.keras.models.load_model(model_dir)
        r = model.evaluate(test_gen)
        print("test loss, test acc:", r)
        #save test results
        np.save("CNN_modelCNR/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size), r)
        plot_history(history.history, epochs, batch_size)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)