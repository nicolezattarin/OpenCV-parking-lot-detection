import visualkeras
import tensorflow as tf

model =tf.keras.models.load_model("CNN_modelCNR/10_epochs_32_batch_classifier")
visualkeras.layered_view(model, to_file='model_layers.png') 
visualkeras.graph_view(model, to_file='model_graph.png') 

