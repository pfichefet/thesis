import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
current_path = os.path.dirname(os.path.abspath(__file__))

"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""

parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str)
# parser.add_argument('-title', type=str)


options = {0: "Outliers",
           1: "MAE",
           2: "MSE",
           3: "MSE",
           4: "MAX",
           }

colorsSkip = {0: "b",
          1: "r",
          2: "y",
          3: "k",
          4: "c",
          }

colorsNorm = {0: "b--",
              1: "r--",
              2: "y--",
              3: "k--",
              4: "c--",
              }

models = {"deepDenoiseSkipSR": "CAeSSC",
           "deepDenoiseNormSR": "NDNN",
           }


# Compute the roc curve and display it onto a graph.
def print_roc_curve(ROC_data, legend):
    i = 0
    opt_list = []
    for curve in ROC_data:
        tp_rate = [0]
        fp_rate = [0]
        for data in curve:
            tp_rate.append(data[2])
            fp_rate.append(data[3])
        print(legend[i].split(".")[0])
        opt = legend[i].split(".")[0][-1]
        if int(opt) not in opt_list:
            opt_list.append(int(opt))
        if legend[i].split("_")[0] == "deepDenoiseSkipSR":
            plt.plot(fp_rate, tp_rate, colorsSkip[int(opt)])#, label="Score = "+options[int(opt)])
        else:
            plt.plot(fp_rate, tp_rate, colorsNorm[int(opt)])#, label="Score = " + options[int(opt)])
        i += 1

    plt.plot([0, 1], [0, 1], 'g')
    split_title = legend[0].split(".")[0][:-1].split("_")
    # title = models[split_title[0]]
    title = split_title[1] #+ "_" + split_title[len(split_title)-2]
    for i in range(2, len(split_title)-1):
        title += "_" + split_title[i]

    b_patch = mpatches.Patch(color='b', label='Score : Outliers')
    r_patch = mpatches.Patch(color='r', label='Score : MAE')
    y_patch = mpatches.Patch(color='y', label='Score : MSE')
    k_patch = mpatches.Patch(color='k', label='Score : MSE')
    c_patch = mpatches.Patch(color='c', label='Score : Max')
    patch_list = [b_patch, r_patch, y_patch, k_patch, c_patch]
    applied_patch = [patch for i,patch in enumerate(patch_list) if i in opt_list]
    plt.legend(handles=applied_patch)
    plt.title(title)
    #plt.legend(loc='best')
    plt.ylabel("True Positive rate")
    plt.xlabel("False Positive rate")
    plt.show()


def load_roc_data(filename):
    with open(current_path+"/obj/ROC/"+filename, 'rb') as output:
        return pickle.load(output)


def main(args):
    file_data = args.file.split(",")
    ROC_data = []
    for data in file_data:
        ROC_data.append(load_roc_data(data))
    print_roc_curve(ROC_data, file_data)


arguments = None
if __name__ == '__main__':
    arguments = parser.parse_args()
main(arguments)
