import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import optimizers


if __name__ == '__main__':
    classes_focal = list(np.arange(40, 501, 10))
    classes_distortion = list(np.arange(0, 61, 1) / 50.)
    n_acc_focal = 0
    n_acc_dist = 0
    with tf.device('/gpu:0'):
        input_shape = (299, 299, 3)
        main_input = Input(shape=input_shape, dtype='float32')
        phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
        phi_features = phi_model.output
        phi_flattened = Flatten()(phi_features)        final_output_focal = Dense(len(classes_focal), activation='softmax')(phi_flattened)        final_output_distortion = Dense(len(classes_distortion), activation='softmax')(        phi_flattened)

        model = Model(inputs=main_input, outputs=[final_output_focal, final_output_distortion]) 
        model.load_weights('models/weights_06_5.61.h5')

        for i in glob.glob('data/*.jpg'):
            image = cv2.imread(i)
            image = cv2.resize(image, (299, 299))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.
            image = image - 0.5
            image = image * 2.
            image = np.expand_dims(image, 0)
            image = preprocess_input(image)
            prediction_focal = model.predict(image)[0]
            prediction_dist = model.predict(image)[1]
            n_acc_focal += classes_focal[np.argmax(prediction_focal[0])]
            n_acc_dist += classes_distortion[np.argmax(prediction_dist[0])]

        print('focal:')
        print(classes_focal[np.argmax(prediction_focal[0])])

        print('dist:')
        print(classes_distortion[np.argmax(prediction_dist[0])])

"""
mtx from classic calibration
array([[1477.45371832,    0.        , 1059.01529907],
       [   0.        , 1475.03552426,  569.81499868],
       [   0.        ,    0.        ,    1.        ]])
"""

