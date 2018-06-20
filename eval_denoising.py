import argparse
from dataset import Dataset
from models_autoencoder import Autoencoder
from models_autoencoder import get_mses
import numpy as np
import keras.optimizers as optimizers
import matplotlib.pyplot as plt


"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str)
parser.add_argument('-kind_class', type=str)
parser.add_argument('-weight', type=str)
parser.add_argument('-residual', type=str)
parser.add_argument('-patch_size', type=int)
parser.add_argument('-batch_size', type=int)
parser.add_argument('--show', type=int, default=0)


def compare_images(img, noisy_img, predict_img):
    plt.figure(figsize=(3, 3))
    # display original
    ax = plt.subplot(3, 3, 1)
    if len(img.shape) == 3 and img.shape[2] == 3:
        plt.imshow(img.reshape(img.shape[0], img.shape[1], img.shape[2]))
    else:
        plt.imshow(img.reshape(img.shape[0], img.shape[1]))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, 3, 2)
    if len(noisy_img.shape) == 3 and noisy_img.shape[2] == 3:
        plt.imshow(noisy_img.reshape(img.shape[0], img.shape[1], img.shape[2]))
    else:
        plt.imshow(noisy_img.reshape(img.shape[0], img.shape[1]))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, 3, 3)
    if len(predict_img.shape) == 3 and predict_img.shape[2] == 3:
        plt.imshow(predict_img.reshape(img.shape[0], img.shape[1], img.shape[2]))
    else:
        plt.imshow(predict_img.reshape(img.shape[0], img.shape[1]))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def main(args):
    dataset = Dataset()
    dataset_noise = Dataset()
    dataset.load_trainset(args.kind_class)
    noise_file = args.kind_class.split(".")[0]+"_gauss_10.npy"
    dataset_noise.load_trainset(noise_file)
    dataset.trainset = dataset.trainset[dataset.trainset.shape[0]-dataset.trainset.shape[0]/8:]
    dataset_noise.trainset = dataset_noise.trainset[dataset_noise.trainset.shape[0]-dataset_noise.trainset.shape[0]/8:]
    init_height = dataset_noise.img_height
    init_width = dataset_noise.img_width
    dataset.patch_dataset(args.patch_size, args.patch_size)
    dataset_noise.patch_dataset(args.patch_size, args.patch_size)
    dataset.pre_processing_image()
    dataset_noise.pre_processing_image()

    input_shape = (args.patch_size, args.patch_size, dataset.img_dim)
    adam = optimizers.Adam(lr=1e-3)
    loss = 'mean_squared_error'
    optimizer = adam
    current_autoencoder = Autoencoder(input_shape, args.model)
    current_autoencoder.model.compile(optimizer=optimizer, loss=loss)
    current_autoencoder.model.load_weights("obj/weight/"+args.weight)
    decoded_imgs = current_autoencoder.model.predict(dataset_noise.trainset, batch_size=args.batch_size)
    noisy_imgs = dataset.rebuild_all_imgs(dataset_noise.trainset,
                                          int(init_height/args.patch_size)*args.patch_size,
                                          int(init_width/args.patch_size)*args.patch_size)
    original_imgs = dataset.rebuild_all_imgs(dataset.trainset,
                                             int(init_height/args.patch_size)*args.patch_size,
                                             int(init_width/args.patch_size)*args.patch_size)
    predict_imgs = dataset.rebuild_all_imgs(decoded_imgs,
                                            int(init_height/args.patch_size)*args.patch_size,
                                            int(init_width/args.patch_size)*args.patch_size)

    if args.residual == "yes":
        predict_imgs = noisy_imgs - predict_imgs
    mses = get_mses(original_imgs, predict_imgs)
    print("mean mse = "+str(np.mean(mses)))

    i = 0
    for id_img,  _ in mses:
        compare_images(original_imgs[id_img], noisy_imgs[id_img], predict_imgs[id_img])
        if i == args.show:
            break
        i += 1

    i = 0
    end_mse = len(mses)-1
    for index in range(end_mse, 0, -1):
        id_img, _ = mses[index]
        compare_images(original_imgs[id_img], noisy_imgs[id_img], predict_imgs[id_img])
        if i == args.show:
            break
        i += 1


arguments = None
if __name__ == '__main__':
    arguments = parser.parse_args()
main(arguments)
