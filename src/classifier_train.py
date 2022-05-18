import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import argparse

def prepare_data(busy_data_path, free_data_path, val_split=0.25, train_split=0.5, test_split=0.25, save=False):
    """
    Prepare data for training and testing. this function could need a lot of time and memory, so it is recommended to run it in a separate process.
    Moreover, the option save=True will save the data in such a way to make it possible to process data with
    tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory

    params:
        busy_data_path: path to busy data
        free_data_path: path to free data
        val_split: percentage of data to be used for validation
        train_split: percentage of data to be used for training
        test_split: percentage of data to be used for testing
    """
    import glob
    import cv2

    if val_split + train_split + test_split != 1:
        raise ValueError("val_split + train_split + test_split must equal 1")
        
    images_busy = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob(busy_data_path)]
    images_free = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob(free_data_path)]
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
    
    train = train_busy + train_free
    val = val_busy + val_free
    test = test_busy + test_free   
    return train, val, test

parser = argparse.ArgumentParser(description='Train a classifier')
parser.add_argument('--epochs', type=int , default=15, help='number of epochs')
parser.add_argument('--batch_size', type=int , default=32, help='batch size')

def main(epochs, batch_size):
    from CNNClassifier import CNNClassifier

    if os.path.exists("../CNN_model/{}_epochs_{}_batch_classifier".format(epochs, batch_size)):
        if input("model already exists, do you want to train again the model? (y/n) if not, the existing model is loaded") == "y":
            model = tf.keras.models.load_model("../CNN_model/{}_epochs_{}_batch_classifier".format(epochs, batch_size))
            history = np.load("CNN_model/{}_epochs_{}_batch_history.npy".format(epochs, batch_size), allow_pickle=True).item()

    else:
        busy_data = "../CNRPark-Patches-150x150/*/busy/*.jpg"
        free_data = "../CNRPark-Patches-150x150/*/free/*.jpg"
        _, _, _ = prepare_data(busy_data, free_data, save=False)

        classifier = CNNClassifier(150,150)
        history = classifier.train(batch_size=batch_size, epochs=epochs)

    sns.set_theme(style="white", font_scale=1.5, palette="Dark2")
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, ax = plt.subplots(figsize=(10,6))
    ax_right=ax.twinx()
    lw=3
    sns.lineplot(epochs, history['accuracy'], lw=lw, label='train', ax=ax)
    sns.lineplot(epochs, history['val_accuracy'], lw=lw, label='validation', ax=ax)
    sns.lineplot(epochs, history['loss'], lw=lw, label='train loss', ax=ax_right)
    sns.lineplot(epochs, history['val_loss'], lw=lw, label='validation loss', ax=ax_right)
    ax.legend(loc='center right')
    ax_right.get_legend().remove()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax_right.set_ylabel('Loss')
    ax.set_ylim(0,1)
    ax_right.set_ylim(0,1)
    fig.savefig("CNN_model/{}_epochs_{}_batch_history.png".format(epochs, batch_size), bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.epochs, args.batch_size)