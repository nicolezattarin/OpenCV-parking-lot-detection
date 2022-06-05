import numpy as np
import seaborn as sns
import glob
import tensorflow as tf

def main():
    from SVMClassifier import SVMClassifier
    busy_data = "../../CNRPark-Patches-150x150/*/busy/*.jpg"
    free_data = "../../CNRPark-Patches-150x150/*/free/*.jpg"

    # Load data
    print("loading data...")
    busy_data_files = np.array(sorted(glob.glob(busy_data)))
    free_data_files = np.array(sorted(glob.glob(free_data)))
    busy_data = []
    free_data = []
    for f in busy_data_files:
        img = tf.keras.utils.load_img(f, target_size=(150, 150))    
        busy_data.append(tf.keras.utils.img_to_array(img)/255.)
    for f in free_data_files:
        img = tf.keras.utils.load_img(f, target_size=(150, 150))    
        free_data.append(tf.keras.utils.img_to_array(img)/255.)
    
    #label 0 for free, 1 for busy
    # prepare data
    X = np.concatenate((busy_data, free_data))
    y = np.concatenate((np.ones(len(busy_data)), np.zeros(len(free_data))))
    
    #shuffle data
    from sklearn.utils import shuffle
    X, y = shuffle(X, y)

    #split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    #train model if we cannot find the file SVMmodel.pkl
    print("Training...")
    import os
    if os.path.isfile('SVMmodel.pkl'):
        print('Model is already trained!')
        model = SVMClassifier()
    else:
        model = SVMClassifier()
        model.train(X_train, y_train, grid_search=True)

    #test model
    print("Testing...")
    test_accuracy = model.evaluate_accuracy(X_test, y_test)
    train_accuracy =  model.evaluate_accuracy(X_train, y_train)
     
    print("test accuracy: ", test_accuracy)
    print("train accuracy: ", train_accuracy)
    
if __name__ == "__main__":
    main()