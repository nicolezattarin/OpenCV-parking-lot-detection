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

def prepare_data(busy_data_path, free_data_path, val_split=0.25, train_split=0.5, test_split=0.25, save=False):
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

    returns:
        train_data, val_data, test_data: prepared data
    """
    import glob
    import cv2

    if val_split + train_split + test_split != 1:
        raise ValueError("val_split + train_split + test_split must equal 1")
        
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

    #save the data if you want to use it later
    if save == True:
        import os 
        if not os.path.isdir("../classifier_data"):
            os.mkdir("../classifier_data")

        if not os.path.isdir("../classifier_data/train"):
            os.mkdir("../classifier_data/train")
            os.mkdir("../classifier_data/train/busy")
            os.mkdir("../classifier_data/train/free")

            print("saving train imgs")
            for i in range(len(train_free)):
                cv2.imwrite("../classifier_data/train/free/"+str(i)+"_train_free.jpg", train_free[i])
            for i in range(len(train_busy)):
                cv2.imwrite("../classifier_data/train/busy/"+str(i)+"_train_busy.jpg", train_busy[i])

        if not os.path.isdir("../classifier_data/test"):
            os.mkdir("../classifier_data/test")
            os.mkdir("../classifier_data/test/busy")
            os.mkdir("../classifier_data/test/free")
            print("saving test imgs")
            for i in range(len(test_free)):
                cv2.imwrite("../classifier_data/test/free/"+str(i)+"_test_free.jpg", test_free[i])
            for i in range(len(test_busy)):
                cv2.imwrite("../classifier_data/test/busy/"+str(i)+"_test_busy.jpg", test_busy[i])

        if not os.path.isdir("../classifier_data/val"):
            print("saving validation imgs")
            os.mkdir("../classifier_data/val")
            os.mkdir("../classifier_data/val/busy")
            os.mkdir("../classifier_data/val/free")
            for i in range(len(val_free)):
                cv2.imwrite("../classifier_data/val/free/"+str(i)+"_val_free.jpg", val_free[i])
            for i in range(len(val_busy)):
                cv2.imwrite("../classifier_data/val/busy/"+str(i)+"_val_busy.jpg", val_busy[i])
    
 
    return train_busy, train_free, val_busy, val_free, test_busy, test_free

def plot_history(history_trend, epochs, batch_size):
    # PLOT HISTORY OF VALIDATION AND TRAINING ACCURACY/LOSS
    sns.set_theme(style="white", font_scale=1.5, palette="Dark2")

    if epochs == 15: 
        epochs_range = range(1, len(history_trend["accuracy"]))
        history_trend['accuracy'], history_trend['val_accuracy'], history_trend['loss'], history_trend['val_loss'] =\
            history_trend['accuracy'][:-1], history_trend['val_accuracy'][:-1], history_trend['loss'][:-1], history_trend['val_loss'][:-1]
    else: epochs_range = range(1, len(history_trend["accuracy"]) +1 )
    
    #load test results 
    if not os.path.exists("CNN_model/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size)):
        print("test data not found")
        return
    test_results = np.load("CNN_model/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size))
    lw=2
    ls='--'
    m = 'o'
    ms = 8
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(epochs_range, history_trend['accuracy'], lw=lw, ls=ls, label='train', ax=ax, marker='o', markersize=ms)
    sns.lineplot(epochs_range, history_trend['val_accuracy'], lw=lw, ls=ls, label='validation', ax=ax, marker='o', markersize=ms)
    ax.legend(loc='lower left')

    # An inner plot to show the peak frequency
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax,  "55%", "60%" ,loc="lower right", borderpad=1)

    sns.lineplot(epochs_range, history_trend['loss'], lw=lw, ls=ls,label='train loss', ax=axins, marker='o', markersize=ms)
    sns.lineplot(epochs_range, history_trend['val_loss'], lw=lw, ls=ls,label='validation loss', ax=axins, marker='o', markersize=ms)

    #dashed lines with accuracy and loss on test set
    ax.axhline(y=test_results[1], label='test', color='black', lw=2, linestyle='--')
    axins.axhline(y=test_results[0], label='test', color='black', lw=2, linestyle='--')

    axins.get_legend().remove()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    axins.set_ylabel('Loss')
    axins.set_xticklabels([])
    axins.set_yticks([0,0.5,1])
    ax.set_ylim(0,1)
    axins.set_ylim(0,1)

    fig.savefig("CNN_model/{}_epochs_{}_batch_history.png".format(epochs, batch_size), bbox_inches='tight')

def write_graph_to_pb(model_dir):
    """
    convert saved_model.pb to frozen graph taken from https://www.ai2news.com/blog/54180/ (in chinese?)
    we need to save a frozen graph to upload the model in opencv c++, otherwise the error
    "'opencv_tensorflow.FunctionDef.Node.ret' contains invalid UTF-8 data when parsing a protocol buffer." occurs
    """
    m = tf.saved_model.load(model_dir)

    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    tfm = tf.function(lambda x: m(x))  # full model
    tfm = tfm.get_concrete_function(tf.TensorSpec(m.signatures['serving_default'].inputs[0].shape.as_list(),
                                                  m.signatures['serving_default'].inputs[0].dtype.name))
    frozen_func = convert_variables_to_constants_v2(tfm)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=model_dir, name="saved_model_graph.pb", as_text=False)

def get_test_generator(test_dir, batch_size):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(directory=test_dir, batch_size=batch_size,
                                                        class_mode='binary', target_size=(150,150))
    return train_generator

parser = argparse.ArgumentParser(description='Train a classifier')
parser.add_argument('--epochs', type=int , default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int , default=32, help='batch size')
parser.add_argument('--load', type=bool , default=False, help='if true an existing model is loaded')
parser.add_argument('--save', type=bool , default=False, help='if truethe images are saved again from zero')

def main(epochs, batch_size, load, save):
    from CNNClassifier import CNNClassifier
    busy_data = "../CNRPark-Patches-150x150/*/busy/*.jpg"
    free_data = "../CNRPark-Patches-150x150/*/free/*.jpg"
    model_dir = "CNN_model/{}_epochs_{}_batch_classifier".format(epochs, batch_size)
    
    if load:
        if not os.path.exists(model_dir): 
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_dir) 

        #load the model
        model = tf.keras.models.load_model(model_dir)
        history = np.load("CNN_model/{}_epochs_{}_batch_history.npy".format(epochs, batch_size), allow_pickle=True).item()

        #frozen model
        write_graph_to_pb(model_dir)
        print("\nmodel loaded")

        #TESTING
        test_gen = get_test_generator("../classifier_data/test", batch_size)
        print("test data prepared")
        r = model.evaluate(test_gen)
        print("test loss, test acc:", r)
        #save test results
        np.save("CNN_model/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size), r)
        plot_history(history, epochs, batch_size)

  
    else:
        if save: _, _, _, _, _, _ = prepare_data(busy_data, free_data, save=True)
        classifier = CNNClassifier(150,150)
        history = classifier.train(batch_size=batch_size, epochs=epochs)
        
        #TESTING
        test_gen = get_test_generator("../classifier_data/test", batch_size)
        print("test data prepared")
        r = model.evaluate(test_gen)
        print("test loss, test acc:", r)
        #save test results
        np.save("CNN_model/{}_epochs_{}_batch_test_results.npy".format(epochs, batch_size), r)
        plot_history(history, epochs, batch_size)
        plot_history(history.history, epochs, batch_size)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)