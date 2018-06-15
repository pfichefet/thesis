# import os
# current_path = os.path.dirname(os.path.abspath(__file__))
#
# models = ["deepDenoiseSkipSR"]
# Classes = ["Class4_256_256"]
# defects = ["gauss"]
# option = ["0","1","2","3","4"]
#
# for clas in Classes:
#     for defect in defects:
#         name = current_path+"/data/train/Load/"+clas+ "_" +defect+ '_10_20.npy'
#         new_name = current_path+"/data/train/Load/"+clas+ "_" + defect+ '_20.npy'
#         print(name)
#         os.rename(name, new_name)

# from dataset import Dataset
# import matplotlib.pyplot as plt
#
#
# def display_image(img):
#     plt.figure(figsize=(2, 2))
#     # display original
#     ax = plt.subplot(2, 2, 1)
#     plt.imshow(img.reshape(img.shape[0], img.shape[1]))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     plt.show()
#
#
# data = Dataset()
# data.load_trainset("Class4_512_512_stain_10.npy")
# data.pre_processing_image()
# data_noise = Dataset()
# data_noise.load_trainset("Class4_256_256_scratch_10.npy")
# data_noise.pre_processing_image()
# for i in range(len(data_noise.trainset)):
#     display_image(data.trainset[i])
    # display_image(data_noise.trainset[0])
    # display_image(data_noise.trainset[0] - data.trainset[0])

import matplotlib.pyplot as plt
plt.plot([0,0.1,1], [0,0.9,1], label="Example 1")
plt.plot([0,0.6,0.8,1], [0,0.3,0.6,1], label="Example 2")
plt.plot([0,0.3,0.6,1], [0,0.6,0.8,1], label="Example 3")
plt.plot([0, 1], [0, 1], 'g--')
title = "ROC curve example"
plt.title(title)
plt.legend(loc='best')
plt.ylabel("True Positive rate")
plt.xlabel("False Positive rate")
plt.show()