import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
current_path = os.path.dirname(os.path.abspath(__file__))


"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""

parser = argparse.ArgumentParser()
parser.add_argument('-files', type=str)

colors = {"res": "b",
          "not_res": "b--",
          }

colorsVal = {"res": "y",
             "not_res": "y--",
             }


def print_history(histories, file_name):
    i = 0
    for history in histories:
        name = file_name[i].split(".")[0]
        name = name.split("_")
        res = "not_res" if name[len(name)-2] == "not" else "res"

        # plt.plot(history['acc'])
        # plt.plot(history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        #
        # plt.plot(history['loss'])
        # plt.plot(history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()

        plt.plot(history['PSNRLoss'], colors[res])
        plt.plot(history['val_PSNRLoss'], colorsVal[res])
        i += 1

    v_patch = mpatches.Patch(color='y', label='Validation')
    nv_patch = mpatches.Patch(color='b', label='Train')
    plt.title('model PSNR')
    plt.ylabel('PSNR')
    plt.xlabel('epoch')
    plt.legend(handles=[nv_patch, v_patch])
    plt.show()


def load_object(filename):
    with open(current_path+"/obj/history/"+filename, 'rb') as output:
        return pickle.load(output)


def main(args):
    file_data = args.files.split(",")
    histories = []
    for data in file_data:
        history_object = load_object(data)
        histories.append(history_object)
    print_history(histories, file_data)


arguments = None
if __name__ == '__main__':
    arguments = parser.parse_args()
main(arguments)
