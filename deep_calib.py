import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import optimizers

with tf.device('/gpu:0'):
    classes_focal = list(np.arange(40, 501, 10))
    classes_distortion = list(np.arange(0, 61, 1) / 50.)
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    final_output_focal = Dense(len(classes_focal), activation='softmax', name='output_focal')(phi_flattened)
    final_output_distortion = Dense(len(classes_distortion), activation='softmax', name='output_distortion')(
        phi_flattened)

    layer_index = 0
    for layer in phi_model.layers:
        layer.name = layer.name + "_phi"

    model = Model(main_input, output=[final_output_focal, final_output_distortion])
    model.load_weights(path_to_weights)

    image = cv2.imread('data/1.jpg')
    image = cv2.resize(image, (299, 299))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = image - 0.5
    image = image * 2.
    image = np.expand_dims(image, 0)
    # image = preprocess_input(image)
