import argparse
import numpy as np
import random
import pickle
from dataset import Dataset
from keras import backend as K
from models_autoencoder import Autoencoder
import keras.optimizers as optimizers
from keras.callbacks import TensorBoard

"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""

parser = argparse.ArgumentParser()
parser.add_argument('-models', type=str)
parser.add_argument('-kind_class', type=str)
parser.add_argument('-noise', type=str)
parser.add_argument('-residual', type=str)
parser.add_argument('-patch_size', type=str)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-epochs', type=int)
parser.add_argument('--gauss_level', type=str, default="")


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def save_object(obj, filename):
    with open("obj/history/"+filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output)


def run_models(models_name, kind_class, noise, residual, patch_size, batch_size, epochs, gauss_level):
    print(models_name, kind_class, noise, residual)
    if models_name == 'deepDenoiseNormSR':
        adam = optimizers.Adam(lr=1e-4)
    else:
        adam = optimizers.Adam(lr=1e-3)
    loss = 'mean_squared_error'
    optimizer = adam
    data = Dataset()
    data_noise1 = Dataset()
    data_noise2 = Dataset()
    name_set = kind_class

    if noise == "gaussian":
        name_noise_set = name_set + "_gauss" + gauss_level + ".npy"
        data_noise1.load_trainset(name_noise_set)
        data_noise2.load_trainset(name_noise_set)
        noise  = noise + gauss_level
    else:
        name_scratch_set = name_set + "_scratch.npy"
        name_stain_set = name_set + "_stain.npy"
        data_noise1.load_trainset(name_scratch_set)
        data_noise2.load_trainset(name_stain_set)

    name_set = name_set + ".npy"
    data.load_trainset(name_set)

    data.trainset = data.trainset[0:data.trainset.shape[0]-data.trainset.shape[0]/8]
    data_noise1.trainset = data_noise1.trainset[0:data_noise1.trainset.shape[0]-data_noise1.trainset.shape[0]/8]
    data_noise2.trainset = data_noise2.trainset[0:data_noise2.trainset.shape[0]-data_noise2.trainset.shape[0]/8]

    c = list(zip(data.trainset, data_noise1.trainset, data_noise2.trainset))
    random.shuffle(c)
    data.trainset, data_noise1.trainset, data_noise2.trainset = zip(*c)
    data.trainset = np.asarray(data.trainset)
    data_noise1.trainset = np.asarray(data_noise1.trainset)
    data_noise2.trainset = np.asarray(data_noise2.trainset)

    data_noise1.trainset = np.append(data_noise1.trainset[0:data.trainset.shape[0]/2],
                                     data_noise2.trainset[data.trainset.shape[0]/2:]).reshape(data_noise1.trainset.shape)

    c = list(zip(data.trainset, data_noise1.trainset))
    random.shuffle(c)
    data.trainset, data_noise1.trainset = zip(*c)
    data.trainset = np.asarray(data.trainset)
    data_noise1.trainset = np.asarray(data_noise1.trainset)

    patch_size_h, patch_size_w = int(patch_size.split(",")[0]), int(patch_size.split(",")[1])
    data.patch_dataset(patch_size_h, patch_size_w)
    data_noise1.patch_dataset(patch_size_h, patch_size_w)

    data.pre_processing_image()
    data_noise1.pre_processing_image()

    if data.image_data_format == 'channels_first':
        input_shape = (data.img_dim, data.img_height, data.img_width)
    else:
        input_shape = (data.img_height, data.img_width, data.img_dim)

    if residual:
        trainset = data_noise1.trainset[0:3*data.trainset.shape[0]/4]
        map_trainset = data_noise1.trainset[0:3*data.trainset.shape[0]/4] - data.trainset[0:3*data.trainset.shape[0]/4]
        validset = data_noise1.trainset[3*data.trainset.shape[0]/4:]
        map_validset = data_noise1.trainset[3*data.trainset.shape[0]/4:] - data.trainset[3*data.trainset.shape[0]/4:]
    else:
        trainset = data_noise1.trainset[0:3*data.trainset.shape[0]/4]
        map_trainset = data.trainset[0:3*data.trainset.shape[0]/4]
        validset = data_noise1.trainset[3*data.trainset.shape[0]/4:]
        map_validset = data.trainset[3*data.trainset.shape[0]/4:]

    autoencoder = Autoencoder(input_shape, models_name)
    autoencoder.model.compile(optimizer=optimizer, loss=loss, metrics=[PSNRLoss, "accuracy"])
    history = autoencoder.model.fit(trainset, map_trainset,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    validation_data=(validset, map_validset),
                                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    residual_name = 'res' if residual else 'not_res'
    kind_class = kind_class.split("_")[0] + "_" + str(patch_size_h) + "_" + str(patch_size_w)
    save_object(history.history, models_name + '_' + kind_class + '_' + noise + '_' + residual_name + ".pkl")
    autoencoder.model.save_weights("obj/weight/"+models_name+'_'+kind_class+'_'+noise+'_'+residual_name+'.h5')


def main(args):
    gauss_level = "" if args.gauss_level == "" else "_"+args.gauss_level
    model_name = args.models.split(",")
    kind_classe = args.kind_class.split(",")
    noises = args.noise.split(",")
    residual = args.residual.split(",")
    for model in model_name:
        for classe in kind_classe:
            for nois in noises:
                for res in residual:
                        run_models(model, classe, nois, int(res), args.patch_size, args.batch_size, args.epochs,
                                   gauss_level)


arguments = None
if __name__ == '__main__':
    arguments = parser.parse_args()
main(arguments)
