import argparse
import numpy as np
from dataset import Dataset

"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""

parser = argparse.ArgumentParser()
parser.add_argument('-load', type=bool, default=False)
parser.add_argument('-train_set', type=str)
parser.add_argument('-test_set', type=str)
parser.add_argument('-save_train_set_name', type=str)
parser.add_argument('-save_test_set_name', type=str)
parser.add_argument('--force_color', type=str, default='random')
parser.add_argument('--downsampled', type=str)
parser.add_argument('--grayscale', type=bool, default=True)
parser.add_argument('--select', type=str)
parser.add_argument('--patch', type=str)
parser.add_argument('--add_defect', type=str)
parser.add_argument('--gauss_level', type=int, default=0)


def main(args):
    dataset = Dataset(args.grayscale)
    if args.train_set:
        if args.load:
            dataset.load_trainset(args.train_set)
        else:
            dataset.create_trainset(args.train_set)
    if args.test_set:
        if args.load:
            dataset.load_testset(args.test_set)
        else:
            dataset.create_testset(args.test_set)
    if args.select:
        cut_str = args.select.split("x")
        height = int(cut_str[0])
        width = int(cut_str[1])
        new_trainset = []
        for img in dataset.trainset:
            if img.shape[0] == height and img.shape[1] == width:
                new_trainset.append(img)
        new_testset = []
        for img in dataset.testset:
            if img.shape[0] == height and img.shape[1] == width:
                new_testset.append(img)
        dataset.img_width = width
        dataset.img_height = height
        dataset.trainset = np.asarray(new_trainset)
        dataset.testset = np.asarray(new_testset)
    if args.patch:
        cut_str = args.patch.split("x")
        height = int(cut_str[0])
        width = int(cut_str[1])
        dataset.patch_dataset(height, width)
    if args.downsampled:
        cut_str = args.downsampled.split("x")
        pixel_shape = (int(cut_str[0]), int(cut_str[1]))
        dataset.downsampled_dataset(pixel_shape)
    if args.add_defect == "gaussian":
        dataset.trainset = dataset.add_gaussian_noise_on_data(dataset.trainset, 1, args.gauss_level)
        dataset.testset = dataset.add_gaussian_noise_on_data(dataset.testset, 1, args.gauss_level)
    elif args.add_defect == "scratch":
        dataset.trainset = dataset.add_scratch_on_data(dataset.trainset, 1, args.force_color)
        dataset.testset = dataset.add_scratch_on_data(dataset.testset, 1, args.force_color)
    elif args.add_defect == "stain":
        dataset.trainset = dataset.add_stain_on_data(dataset.trainset, 1)
        dataset.testset = dataset.add_stain_on_data(dataset.testset, 1)
    if dataset.trainset.any():
        dataset.save_trainset(args.save_train_set_name)
        print("trainset - "+args.save_train_set_name+" created.")
    if dataset.testset.any():
        dataset.save_testset(args.save_test_set_name)
        print("testset - " + args.save_test_set_name + " created.")


args = None
if __name__ == '__main__':
    args = parser.parse_args()
main(args)
