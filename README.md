import tensorflow as tf
from tensorflow import keras
import numpy as np
fashion_mnist-keras.datasets.fashion_mnist
(train_images, train_labels), (test images, test labels)-fashion_mnist.load_data()
train_images=train_images/255.0
test_images=test_images/255.0
train images[0].shape
(28, 28)
train_images train_images.reshape(len(train_images),28,28,1)
test images test_images.reshape(len(test_images),28,28,1)
def build_model(hp):
model keras.Sequential([
keras.layers.Conv2D(
filters-hp. Int('conv_1_filter, min_value=32, max_value-128, step-16), kernel_size=hp.Ch
activation='relu',
input_shape=(28,28,1)
), keras.layers.Conv2D(
filters-hp.Int( 'conv_2_filter, min value-32, max value-64, step-16),
kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
activation='relu'
keras.layers.Flatten(),
keras.layers.Dense(
units-hp. Int('dense 1 units', min_value-32, max_value-128, step-16),
activation='relu'
keras.layers.Dense(10, activation='softmax) foutput layer
model.compile(optimizer-keras.optimizers.Adam(hp.Choice('learning rate', values-[le-2, 1
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
return model
from kerastuner import Random Search
from kerastuner.engine.hyperparameters import
tuner_search-RandomSearch(build_model,
File

+ Code
[ ]
X
Comment
Connect -
Share
Editing
HyperParameters
objective= val_accuracy,
max_trials-5, directory='output,project_name="Mnist Fashion")
INFO: tensorflow: Reloading Oracle from existing project output/Mnist Fashion/oracle.jso
tuner_search.search(train_images,train_labels, epochs-3, validation_split-8.1)
INFO:tensorflow:Oracle triggered exit
model-tuner_search.get_best_models(num_models=1)[0]
model.summary()
