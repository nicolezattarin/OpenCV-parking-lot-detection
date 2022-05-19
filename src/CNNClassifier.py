from warnings import WarningMessage
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

class CNNClassifier():

    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
        self.trained = False

    def _get_generators(self, train_directory, val_directory, batch_size):

        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=60,
                                        width_shift_range=0.2, height_shift_range=0.2,
                                        shear_range=0.2, zoom_range=0.2,
                                        horizontal_flip=True, fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(directory=train_directory,
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            target_size=(self.img_height, self.img_width))

        validation_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, 
                                            width_shift_range=0.2, height_shift_range=0.2, 
                                            shear_range=0.2, zoom_range=0.2, 
                                            horizontal_flip=True, fill_mode='nearest')

        validation_generator = validation_datagen.flow_from_directory(directory=val_directory,
                                                                    batch_size=batch_size,
                                                                    class_mode='binary',
                                                                    target_size=(self.img_height, self.img_width))
        return train_generator, validation_generator

    def _create_model(self, img_height, img_width, dropout=0.5):
        # shape note: if you have 30 images of 50x50 pixels in RGB (3 channels), 
        # the shape of your input data is (30,50,50,3).

        input_shape = (img_height, img_width, 3)
        model = Sequential([ 
                            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
                            MaxPooling2D(),
                            Conv2D(32, (3,3), activation='relu'),
                            MaxPooling2D(),
                            Conv2D(64, (3,3), activation='relu'),
                            MaxPooling2D(),
                            Dropout(dropout),
                            Flatten(),
                            Dense(128, activation='relu'),
                            Dense(1, activation='sigmoid')
                            ])

        model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy']) 
        print("Model created \nSUMMARY:")
        print(model.summary())
        return model

    def train(self, train_dir="../classifier_data/train" , val_dir="../classifier_data/val",
                     dropout=0.5, batch_size=32, epochs=15):

        if self.trained == True: 
            raise Exception("Model has already been trained.")

        img_height, img_width = self.img_width, self.img_width
        self.model = self._create_model(img_height, img_width)
        train_generator, validation_generator = self._get_generators(train_dir, val_dir, batch_size=batch_size)
        self.history = self.model.fit(train_generator, epochs=epochs, verbose=1, validation_data=validation_generator)

        self.trained = True
        import os
        if not os.path.isdir("CNN_model"): os.mkdir("CNN_model")
        model_dir = "CNN_model/{}_epochs_{}_batch_classifier".format(epochs, batch_size)
        self._save_model(model_name=model_dir)
        self._write_graph_to_pb(model_dir=model_dir)
        self._save_history(history_name="CNN_model/{}_epochs_{}_batch_history.npy".format(epochs, batch_size))
        return self.history

    def _save_model(self, model_name="classifier"):
        if self.trained:
            self.model.save(model_name, save_format='tf')
        else:
            print("Model has not been trained yet.")

    def _save_history(self, history_name="history"):
        if self.trained:
            np.save(history_name, self.history.history)
        else:
            print("Model has not been trained yet.")

    def load_model(self, model_name="classifier"):
        if self.trained:
            self.model = tf.keras.models.load_model(model_name)
        else:
            print("Model has not been trained yet.")

    def _write_graph_to_pb(self, model_dir="CNN_model"):
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
