import argparse
import matplotlib.pyplot as plt
from dataset import Dataset

"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""

parser = argparse.ArgumentParser()
parser.add_argument('-dataset1', type=str)
parser.add_argument('-dataset2', type=str)
parser.add_argument('--limit', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)


# Input: Takes two numpy images.
# Display the two images side by side.
def compare_images(img, mod_img):
    plt.figure(figsize=(2, 2))
    # display original
    ax = plt.subplot(2, 2, 1)
    plt.imshow(img.reshape(img.shape[0], img.shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, 2, 2)
    plt.imshow(mod_img.reshape(img.shape[0], img.shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


# Input:    original = a list of numpy images. predicted = a list of images.
#           start_id = The images from which we start displaying. Limit = number of images that we want to display.
# Images in original and its corresponding image in predicted are displayed, by couple of two.
def compare_list_images(original, predicted, start_id, limit=10):
    i = 0
    for img_id in range(start_id, len(original)-1):
        if img_id > len(predicted):
            break
        compare_images(original[img_id], predicted[img_id])
        if i == limit:
            break
        i += 1


def main(args):
    dataset1 = Dataset(True)
    dataset2 = Dataset(True)
    dataset1.load_trainset(args.dataset1)
    dataset2.load_trainset(args.dataset2)
    compare_list_images(dataset1.trainset, dataset2.trainset, args.start_id, args.limit)


arguments = None
if __name__ == '__main__':
    arguments = parser.parse_args()
main(arguments)
