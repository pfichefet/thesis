import argparse
from dataset import Dataset
from models_autoencoder import get_residual_mses, get_score, compute_mse, compute_l0, compute_l1, compute_max_error
from models_autoencoder import Autoencoder
import random
import pickle
import keras.optimizers as optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
current_path = os.path.dirname(os.path.abspath(__file__))


"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""

parser = argparse.ArgumentParser()
parser.add_argument('-models', type=str)
parser.add_argument('-kind_class', type=str)
parser.add_argument('-weights', type=str)
parser.add_argument('-patch_size', type=str)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-option', type=str)
parser.add_argument('-thres_mean', type=int)
parser.add_argument('-thres_search', type=int)
parser.add_argument('-perf_mean', type=int)
parser.add_argument('--ROC', type=bool, default=False)
parser.add_argument('--show', type=str, default="no")
parser.add_argument('--write', type=str, default="no")


class Performance():
    def __init__(self, autoencoder, clean_dataset, noisy_dataset, crop_size, batch_size, option, residual):
        self.clean_dataset = clean_dataset
        self.noisy_dataset = noisy_dataset
        self.img_height = clean_dataset.shape[1]
        self.img_width = clean_dataset.shape[2]
        self.crop_img_height = int(crop_size.split(",")[0])
        self.crop_img_width = int(crop_size.split(",")[1])
        self.img_dim = clean_dataset.shape[3] if len(clean_dataset.shape) == 4 else 1
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.option = option
        self.residual = residual

    def compute_random_dataset(self, size_dataset):
        if size_dataset < self.clean_dataset.shape[0]:
            raise ValueError('The dataset size must be greater or equal to the number of clean images given.')
        number_of_clean_imgs = np.random.randint(self.clean_dataset.shape[0] / 3, self.clean_dataset.shape[0])
        number_of_noisy_imgs = size_dataset - number_of_clean_imgs
        clean_imgs_seleted = np.random.choice(self.clean_dataset.shape[0], number_of_clean_imgs)
        noisy_imgs_selected = np.random.choice(self.noisy_dataset.shape[0], number_of_noisy_imgs)
        clean_testset = self.clean_dataset[clean_imgs_seleted]
        noisy_testset = self.noisy_dataset[noisy_imgs_selected]

        testset = np.append(clean_testset, noisy_testset).reshape((clean_testset.shape[0] + noisy_testset.shape[0],
                                                                   clean_testset.shape[1], clean_testset.shape[2], self.img_dim))
        labels = np.zeros((testset.shape[0]))
        labels[number_of_clean_imgs:] = 1
        labels = labels.tolist()

        c = list(zip(testset, labels))
        random.shuffle(c)
        testset, labels = zip(*c)
        testset = np.asarray(testset)

        data = Dataset()
        data.testset = testset
        data.test_size = testset.shape[0]
        data.img_width = testset.shape[2]
        data.img_height = testset.shape[1]
        data.img_dim = testset.shape[3]
        data.patch_dataset(self.crop_img_height, self.crop_img_width)
        data.testset = data.testset.reshape((data.testset.shape[0], self.crop_img_height, self.crop_img_width, self.img_dim))
        return data, labels, number_of_noisy_imgs

    def compute_threshold(self, num_of_mean, dataset_size):
        threshold = 0
        performance = 0
        for i in range(num_of_mean):
            data, labels, number_of_noisy_imgs = self.compute_random_dataset(dataset_size)
            original_imgs, predict_imgs, threshold_init = self.compute_prediction(data, number_of_noisy_imgs, i == 0)
            threshold = threshold_init if i == 0 else threshold
            defect_ids = self.option_find_outliers(original_imgs, predict_imgs, threshold)
            result = self.autoencoder.performance(defect_ids, labels)
            performance += result[0]
        return threshold, performance/num_of_mean

    def search_threshold(self, num_of_search, num_of_mean, dataset_size):
        list_threshold = []
        for i in range(num_of_search):
            list_threshold.append(self.compute_threshold(num_of_mean, dataset_size))
        return sorted(list_threshold, key=lambda x: x[1], reverse=True)

    def compute_prediction(self, dataset, number_of_noisy_imgs, compute_threshold=False):
        decoded_imgs = self.autoencoder.model.predict(dataset.testset, batch_size=self.batch_size)
        original_imgs = dataset.rebuild_all_imgs(dataset.testset, self.img_height, self.img_width)  # need to change something here
        predict_imgs = dataset.rebuild_all_imgs(decoded_imgs, self.img_height, self.img_width)
        threshold = 0
        if compute_threshold:
            big_mses = self.find_score(original_imgs, predict_imgs)
            threshold = big_mses[number_of_noisy_imgs][1]
        return original_imgs, predict_imgs, threshold

    def test_performance(self, threshold, num_of_mean,  dataset_size, show=False):
        accuracy = 0
        precision = 0
        recall = 0
        specificity = 0
        for i in range(num_of_mean):
            data, labels, number_of_noisy_imgs = self.compute_random_dataset(dataset_size)
            original_imgs, predict_imgs, _ = self.compute_prediction(data, number_of_noisy_imgs)
            defect_ids = self.option_find_outliers(original_imgs, predict_imgs, threshold)
            if show:
                print(labels)
                compare_list_images(defect_ids, original_imgs, predict_imgs, original_imgs-predict_imgs, 30)
            result = self.autoencoder.performance(defect_ids, labels)
            accuracy += result[0]
            precision += result[1]
            recall += result[2]
            specificity += result[3]
        return accuracy/num_of_mean, precision/num_of_mean, recall/num_of_mean, specificity/num_of_mean

    def get_roc_curve(self, dataset_size):
        roc_curve_data = []
        data, labels, number_of_noisy_imgs = self.compute_random_dataset(dataset_size)
        original_imgs, predict_imgs, _ = self.compute_prediction(data, number_of_noisy_imgs)
        scores = self.find_score(original_imgs, predict_imgs)
        defect_ids = []
        # print(labels)
        for defect_id, threshold in scores:
            defect_ids.append(defect_id)
            # compare_list_images(defect_ids, original_imgs, predict_imgs, original_imgs - predict_imgs, 30)
            result = list(self.autoencoder.performance(defect_ids, labels))
            result.append(threshold)
            # print(result)
            roc_curve_data.append(result)
        return roc_curve_data

    def option_find_outliers(self, original_imgs, predict_imgs, threshold):
        scores = self.find_score(original_imgs, predict_imgs)
        defect_ids = self.autoencoder.detect_outliers(scores, threshold)
        return defect_ids

    def find_score(self, original_imgs, predict_imgs):
        if self.residual:
            residual = predict_imgs
            denoise_original = original_imgs - predict_imgs
        else:
            residual = original_imgs - predict_imgs
            denoise_original = predict_imgs
        if self.option == 4:
            scores = get_score(original_imgs, denoise_original, compute_max_error)
        elif self.option == 3:
            scores = get_score(original_imgs, denoise_original, compute_mse)
        elif self.option == 2:
            scores = get_residual_mses(residual)
        elif self.option == 1:
            scores = get_score(original_imgs, denoise_original, compute_l1)
        elif self.option == 0:
            scores = get_score(original_imgs, denoise_original, compute_l0)
        else:
            raise ValueError('Invalid option')
        return scores


def save_roc_curve_data(obj, filename):
    with open(current_path+"/obj/ROC/"+filename, 'w+') as output:  # Overwrites any existing file.
        pickle.dump(obj, output)


def write_performance_result(line_result, outfile_name):
    for information in line_result:
        with open(outfile_name, "a") as myfile:
            myfile.write(information)
            myfile.write(", ")
    with open(outfile_name, "a") as myfile:
        myfile.write("\n")


def compare_images(img, mod_img1, mod_img2):
    plt.figure(figsize=(2, 2))
    # display original
    ax = plt.subplot(1, 3, 1)
    plt.imshow(img.reshape(img.shape[0], img.shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(1, 3, 2)
    plt.imshow(mod_img1.reshape(img.shape[0], img.shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(1, 3, 3)
    plt.imshow(mod_img2.reshape(img.shape[0], img.shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def compare_list_images(img_ids, original, predicted, residual, limit=10):
    i = 0
    for img_id in img_ids:
        print(img_id)
        compare_images(original[img_id], predicted[img_id], residual[img_id])
        if i == limit:
            break
        i += 1


def main(args):
    args.show = True if args.show == "yes" else False
    list_of_models = args.models.split(",")
    list_of_class = args.kind_class.split(",")
    list_of_weight = args.weights.split(",")
    list_of_option = args.option.split(",")
    for class_name in list_of_class:
        for current_models_name in list_of_models:
            for weight in list_of_weight:
                for option in list_of_option:
                    print(weight, current_models_name, class_name, option)
                    patch_size = args.patch_size.split(",")
                    patch_size = patch_size[0] + "_" + patch_size[1]
                    weight_models = current_models_name + "_" + class_name.split("_")[0] + "_" + patch_size + weight
                    dataset_test_name = class_name
                    build_datatset_train = class_name.split("_")
                    dataset_train_name = build_datatset_train[0] + "_" + build_datatset_train[1] + "_" + build_datatset_train[2].split(".")[0] + ".npy"
                    split_type_defect = class_name.split("_")
                    type_defect = split_type_defect[3].split(".")[0] + "_" if len(split_type_defect) == 4 else ""
                    save_result_name = weight_models.split(".")[0] + "_" + type_defect + option
                    residual = True if len(weight.split("_")) == 3 else False

                    data = Dataset()
                    data.load_trainset(dataset_train_name)
                    data.load_testset(dataset_test_name)
                    data.pre_processing_image()

                    start_valset = data.trainset.shape[0]-data.trainset.shape[0]/8
                    length_valset = (data.trainset.shape[0]-start_valset) / 2
                    clean_val_dataset = data.trainset[start_valset:start_valset+length_valset]
                    noisy_val_dataset = data.testset[0:data.testset.shape[0]/3]
                    clean_test_dataset = data.trainset[start_valset+length_valset:]
                    noisy_test_dataset = data.testset[data.testset.shape[0]/3:]

                    patch_size_h, patch_size_w = int(patch_size.split("_")[0]), int(patch_size.split("_")[1])
                    input_shape = (patch_size_h, patch_size_w, 1)
                    if current_models_name == "deepDenoiseNormSR":
                        adam = optimizers.Adam(lr=1e-4)
                    else:
                        adam = optimizers.Adam(lr=1e-3)
                    loss = 'mean_squared_error'
                    optimizer = adam
                    current_autoencoder = Autoencoder(input_shape, current_models_name)
                    current_autoencoder.model.compile(optimizer=optimizer, loss=loss)
                    current_autoencoder.model.load_weights(current_path+"/obj/weight/"+weight_models)

                    evaluate_threshold = Performance(current_autoencoder, clean_val_dataset, noisy_val_dataset,
                                                     args.patch_size, args.batch_size, int(option), residual)
                    evaluate_perf = Performance(current_autoencoder, clean_test_dataset, noisy_test_dataset,
                                                args.patch_size, args.batch_size, int(option), residual)
                    evaluate_roc_curve = Performance(current_autoencoder, data.trainset[start_valset:], data.testset,
                                                     args.patch_size, args.batch_size, int(option), residual)
                    if args.ROC:
                        size_dataset = evaluate_roc_curve.clean_dataset.shape[0] + evaluate_roc_curve.clean_dataset.shape[0] / 5
                        roc_curve_data = evaluate_roc_curve.get_roc_curve(size_dataset)
                        filename = save_result_name + ".pkl"
                        save_roc_curve_data(roc_curve_data, filename)
                    else:
                        value = evaluate_threshold.search_threshold(args.thres_search, args.thres_mean,
                                                                    clean_val_dataset.shape[0] +
                                                                    clean_val_dataset.shape[0] / 5)
                        best_threshold = value[0][0]
                        performance = evaluate_perf.test_performance(best_threshold, args.perf_mean,
                                                                     clean_test_dataset.shape[0] +
                                                                     clean_test_dataset.shape[0] / 5,
                                                                     show=args.show)
                        result = [save_result_name, "option: "+str(option), "accuracy: "+str(performance[0]), "precision: "+str(performance[1]), "recall: "+str(performance[2]), "Specificity: "+str(performance[3])]
                        print(result)
                        if args.write == "yes":
                            write_performance_result(result, current_path+"/performance_models.txt")


arguments = None
if __name__ == '__main__':
    arguments = parser.parse_args()
main(arguments)
