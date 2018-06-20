from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization, Activation, Add
import keras.optimizers as optimizers
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import math

"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""


class Autoencoder():
    def __init__(self, models_input_shape, model_name):
        self.model_input_shape = models_input_shape
        self.model_name = model_name
        self.model = self.load_model()

    def convolutional_autoencoder(self):
        input_img = Input(self.model_input_shape)
        x = Conv2D(128, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(64, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(1, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def deepDenoiseSR(self):
        input_img = Input(self.model_input_shape)
        c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Conv2D(self.model_input_shape[2], (3, 3), activation='linear', padding='same')(m2)

        autoencoder = Model(input_img, decoded)
        # adam = optimizers.Adam(lr=1e-3)
        # model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        return autoencoder

    def deepDenoiseSkipSR(self):
        input_img = Input(self.model_input_shape)
        c1 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_img)
        c2 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(c1)
        c3 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(c2)
        c4 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(c3)
        c5 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(c4)
        c6 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(c5)

        d6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
        m6 = Add()([c6, d6])
        d6 = UpSampling2D()(m6)
        d5 = Conv2D(64, (3, 3), activation='relu', padding='same')(d6)
        d5 = UpSampling2D()(d5)
        d4 = Conv2D(64, (3, 3), activation='relu', padding='same')(d5)
        m4 = Add()([c4, d4])
        d4 = UpSampling2D()(m4)
        d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(d4)
        d3 = UpSampling2D()(d3)
        d2 = Conv2D(64, (3, 3), activation='relu', padding='same')(d3)
        m2 = Add()([c2, d2])
        d2 = UpSampling2D()(m2)
        d1 = Conv2D(64, (3, 3), activation='relu', padding='same')(d2)
        d1 = UpSampling2D()(d1)

        decoded = Conv2D(self.model_input_shape[2], (3, 3), activation='linear', padding='same')(d1)
        mdecoded = Add()([input_img, decoded])

        autoencoder = Model(input_img, mdecoded)
        # adam = optimizers.Adam(lr=1e-3)
        # model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        return autoencoder

    def deepDenoiseNormSR(self):
        input_img = Input(self.model_input_shape)
        c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)

        c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        x = BatchNormalization()(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c3)

        c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c4)

        c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c5)

        c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c6)

        c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c7)

        c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c8)

        c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c9)

        c10 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c10)

        c11 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c11)

        c12 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c12)

        c13 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c13)

        c14 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c14)

        c15 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c15)

        c16 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c16)

        c17 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c17)

        c18 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c18)

        c19 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(c19)

        decoded = Conv2D(self.model_input_shape[2], (3, 3), activation='linear', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def load_model(self):
        if self.model_name == 'convolutional_autoencoder':
            return self.convolutional_autoencoder()
        elif self.model_name == 'deepDenoiseSR':
            return self.deepDenoiseSR()
        elif self.model_name == 'deepDenoiseNormSR':
            return self.deepDenoiseNormSR()
        elif self.model_name == 'deepDenoiseSkipSR':
            return self.deepDenoiseSkipSR()
        else:
            raise ValueError('Unknown model name %s was given' % self.model_name)

    @staticmethod
    def performance(defect_ids, labels):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        i = 0
        for label in labels:
            if label == 0:
                if i not in defect_ids:
                    tn += 1
                else:
                    fp += 1
            elif label == 1:
                if i in defect_ids:
                    tp += 1
                else:
                    fn += 1
            i += 1
        precision = float(tp)/(tp+fp) if tp+fp != 0 else 0
        recall = float(tp)/(tp+fn) if tp+fn != 0 else 0
        accuracy = float(tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn != 0 else 0
        falsePositiveRate = float(fp)/(fp + tn) if fp + tn != 0 else 0
        return accuracy, precision, recall, falsePositiveRate

    @staticmethod
    def detect_outliers(scores, threshold):
        outliers = []
        for x in scores:
            if x[1] >= threshold:
                outliers.append(x[0])
            else:
                break
        return outliers


def build_pure_mask(img):
    mean_list = []
    for i in range(img.shape[0]):
        mean_list.append(np.mean(img[i, :]))
    distance_mean = []
    for mean in mean_list:
        distance = 0
        for mean2 in mean_list:
            distance += math.fabs(mean - mean2)
        distance_mean.append(distance)
    min_distance = min(distance_mean)
    position = 0
    for distance in distance_mean:
        if distance == min_distance:
            break
        position += 1
    value = mean_list[position]
    return np.full(img.shape, value)


def get_residual_mses(prediction):
    n = len(prediction)
    list_mse = {}
    for i in range(n):
        pure_mask = build_pure_mask(prediction[i])
        mse = compute_mse(pure_mask, prediction[i], 255)
        list_mse[i] = mse
    return [(k, list_mse[k]) for k in sorted(list_mse, key=list_mse.get, reverse=True)]


def compute_mse(input_img, predict_img, factor):
    mse = 0
    for pixel1 in range(input_img.shape[0]):
        for pixel2 in range(input_img.shape[1]):
            pre_mse = input_img[pixel1, pixel2]*factor - predict_img[pixel1, pixel2]*factor
            mse += math.pow(pre_mse, 2)
    return mse/(input_img.shape[0]*input_img.shape[1])


def compute_max_error(input_img, predict_img, factor):
    maximum_value = 0
    for pixel1 in range(input_img.shape[0]):
        for pixel2 in range(input_img.shape[1]):
            value = abs(input_img[pixel1, pixel2] * factor - predict_img[pixel1, pixel2] * factor)
            if value > maximum_value:
                maximum_value = value
    return maximum_value


def compute_l1(input_img, predict_img, factor):
    error = 0
    for pixel1 in range(input_img.shape[0]):
        for pixel2 in range(input_img.shape[1]):
            error += abs(input_img[pixel1, pixel2] * factor - predict_img[pixel1, pixel2] * factor)
    return error/(input_img.shape[0]*input_img.shape[1])


def compute_l0(input_img, predict_img, factor):
    errors = []
    for pixel1 in range(input_img.shape[0]):
        for pixel2 in range(input_img.shape[1]):
            errors.append(abs(input_img[pixel1, pixel2] * factor - predict_img[pixel1, pixel2] * factor))
    mean_error = sum(errors)/(input_img.shape[0]*input_img.shape[1])
    sd = 0
    for err in errors:
        sd += math.pow(err - mean_error, 2)
    sd = sd/(len(errors)-1)
    sd = math.sqrt(sd)
    outliers = 0
    for err in errors:
        if err >= 2*sd:
            outliers += 1
    return outliers


def get_score(test, prediction, compute_score):
    n = len(prediction)
    list_error = {}
    for i in range(n):
        error = compute_score(test[i], prediction[i], 255)
        list_error[i] = error
    return [(k, list_error[k]) for k in sorted(list_error, key=list_error.get, reverse=True)]
