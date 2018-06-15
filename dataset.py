import os
import cv2
import numpy as np
from keras import backend as K
import math
current_path = os.path.dirname(os.path.abspath(__file__))

"""Fichefet Pierrick 26631000"""
"""MASTER THESIS"""


class Dataset():
    def __init__(self, grayscale=True):
        self.grayscale = grayscale
        self.image_data_format = K.image_data_format()
        self.folder_train = ""
        self.folder_test = ""
        self.trainset = np.empty((0, 0))
        self.testset = np.empty((0, 0))
        self.train_size = 0
        self.test_size = 0
        self.img_width = 0
        self.img_height = 0
        self.img_dim = 0

    def create_trainset(self, folder_path, size=None):
        self.trainset = self.create_dataset(current_path+"/data/train/" + folder_path, size)
        self.train_size = self.trainset.shape[0]
        self.img_width = self.trainset.shape[2]
        self.img_height = self.trainset.shape[1]
        self.img_dim = 1
        if len(self.trainset.shape) == 4:
            self.img_dim = self.trainset.shape[3]
        self.folder_train = folder_path.split(".")[len(folder_path.split("."))-2]

    def create_testset(self, folder_path, size=None):
        self.testset = self.create_dataset(current_path+"/data/test/" + folder_path, size)
        self.test_size = self.testset.shape[0]
        self.img_width = self.testset.shape[2]
        self.img_height = self.testset.shape[1]
        self.img_dim = 1
        if len(self.testset.shape) == 4:
            self.img_dim = self.testset.shape[3]
        self.folder_test = folder_path.split(".")[len(folder_path.split("."))-2]

    def load_trainset(self, folder_path):
        self.trainset = np.load(current_path+"/data/train/Load/"+folder_path)
        self.train_size = self.trainset.shape[0]
        self.img_width = self.trainset.shape[2]
        self.img_height = self.trainset.shape[1]
        self.img_dim = 1
        if len(self.trainset.shape) == 4:
            self.img_dim = self.trainset.shape[3]
        self.folder_train = folder_path.split(".")[len(folder_path.split("."))-2]

    def load_testset(self, folder_path):
        # if self.trainset.any():
        #     print('You should load a trainset before a testset, the images in the testset and the trainset should '
        #           'have the same size.')
        self.testset = np.load(current_path+"/data/test/Load/"+folder_path)
        self.test_size = self.testset.shape[0]
        self.img_width = self.testset.shape[2]
        self.img_height = self.testset.shape[1]
        self.img_dim = 1
        if len(self.trainset.shape) == 4:
            self.img_dim = self.trainset.shape[3]
        folder_path = folder_path.split(".")[len(folder_path.split("."))-2]
        self.folder_test = folder_path.split(".")[len(folder_path.split("."))-2]

    def save_trainset(self, folder_name=""):
        if not self.trainset.any():
            raise ValueError('The train set is not load.')
        if folder_name == "":
            np.save(current_path+"/data/train/Load/"+self.folder_train, self.trainset)
        else:
            np.save(current_path+"/data/train/Load/" + folder_name.split(".")[len(folder_name.split("."))-2], self.trainset)

    def save_testset(self, folder_name=""):
        if not self.testset.any():
            raise ValueError('The test set is not load.')
        if folder_name == "":
            np.save(current_path+"/data/test/Load/"+self.folder_test, self.testset)
        else:
            np.save(current_path+"/data/test/Load/" + folder_name.split(".")[len(folder_name.split("."))-2], self.testset)

    def create_dataset(self, folder_path, size):
        list_file_path = []
        images = []
        num_imgs = 0
        for filename in os.listdir(folder_path):
            list_file_path.append(os.path.join(folder_path, filename))
        for path in sorted(list_file_path):
            img = cv2.imread(path)
            if img is not None:
                if self.grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img)
                num_imgs += 1
            if num_imgs == size:
                break
        return np.asarray(images)

    def pre_processing_image(self):
        if self.trainset.any():
            self.trainset = process_image(self.trainset, self.img_dim, self.img_height, self.img_width,
                                          self.image_data_format)
        if self.testset.any():
            self.testset = process_image(self.testset, self.img_dim, self.img_height, self.img_width,
                                         self.image_data_format)

    def patch_dataset(self, height, width):
        if height > self.img_height:
            raise ValueError('The height is higher than the height of the original image.')
        if width > self.img_width:
            raise ValueError('The width is higher than the width of the original image.')
        if height < 0 or width < 0:
            raise ValueError('The width and height must be greater than zero.')
        if height == self.img_height and width == self.img_width:
            return self
        st_y = 0 if self.img_height % height == 0 else (self.img_height % height)/2
        st_x = 0 if self.img_width % width == 0 else (self.img_width % width)/2
        end_y = self.img_height - 1 - (self.img_height % height - (self.img_height % height)/2)
        end_x = self.img_width - 1 - (self.img_width % width - (self.img_width % width) / 2)
        set_dataset = {'train': self.trainset, 'test': self.testset}
        for data_id in ['train', 'test']:
            dataset = set_dataset[data_id]
            if dataset.any():
                i = 0
                new_dataset = np.empty(
                    (int(self.img_height / height) * int(self.img_width / width) * dataset.shape[0], height, width, self.img_dim))
                for img in dataset:
                    st_y2 = st_y
                    while st_y2 < end_y:
                        st_x2 = st_x
                        while st_x2 < end_x:
                            path_img = cut_sub_img(img, (st_y2, st_x2), height, width)
                            st_x2 += width
                            new_dataset[i] = path_img
                            i += 1
                        st_y2 += height
                if data_id == 'train':
                    self.trainset = new_dataset
                    self.train_size = new_dataset.shape[0]
                else:
                    self.testset = new_dataset
                    self.test_size = new_dataset.shape[0]
        if self.trainset.any() or self.testset.any():
            self.img_height = height
            self.img_width = width
            self.folder_train = self.folder_train + "_patch"
            self.folder_test = self.folder_test + "_patch"

    def downsampled_dataset(self, pixel_shape):
        new_height = 0
        new_width = 0
        set_dataset = {'train': self.trainset, 'test': self.testset}
        for data_id in ['train', 'test']:
            dataset = set_dataset[data_id]
            if dataset.any():
                new_height = dataset.shape[1] / pixel_shape[0]
                new_width = dataset.shape[2] / pixel_shape[1]
                i = 0
                new_dataset = np.empty((dataset.shape[0], new_height, new_width, self.img_dim))
                for img in dataset:
                    downsampled_img = downsampled(img, pixel_shape)
                    downsampled_img = downsampled_img.reshape(new_height, new_width, self.img_dim)
                    new_dataset[i] = downsampled_img
                    i += 1
                if data_id == 'train':
                    self.trainset = new_dataset
                    self.train_size = new_dataset.shape[0]
                else:
                    self.testset = new_dataset
                    self.test_size = new_dataset.shape[0]
        if self.trainset.any() or self.testset.any():
            self.img_height = new_height
            self.img_width = new_width
            self.folder_train = self.folder_train + "_downsampled"
            self.folder_test = self.folder_test + "_downsampled"

    @staticmethod
    def add_scratch_on_data(dataset, proportion, force_color):
        num_of_imgs = int(dataset.shape[0] * proportion)
        modified_data = dataset[dataset.shape[0]-num_of_imgs:]
        datatrain_1 = np.copy(dataset)[0:dataset.shape[0]-num_of_imgs]
        datatrain_2 = np.empty(modified_data.shape)
        i = 0
        for img in modified_data:
            img_modified = add_scratch(img, force_color)
            datatrain_2[i] = img_modified
            i += 1
        return np.append(datatrain_1, datatrain_2).reshape(dataset.shape)

    @staticmethod
    def add_stain_on_data(dataset, proportion):
        num_of_imgs = int(dataset.shape[0] * proportion)
        modified_data = dataset[dataset.shape[0] - num_of_imgs:]
        datatrain_1 = np.copy(dataset)[0:dataset.shape[0] - num_of_imgs]
        datatrain_2 = np.empty(modified_data.shape)
        i = 0
        for img in modified_data:
            img_modified = add_stain(img)
            datatrain_2[i] = img_modified
            i += 1
        return np.append(datatrain_1, datatrain_2).reshape(dataset.shape)


    @staticmethod
    def add_gaussian_noise_on_data(dataset, proportion, level):
        num_of_imgs = int(dataset.shape[0] * proportion)
        modified_data = dataset[dataset.shape[0] - num_of_imgs:]
        datatrain_1 = np.copy(dataset)[0:dataset.shape[0] - num_of_imgs]
        datatrain_2 = np.empty(modified_data.shape)
        i = 0
        for img in modified_data:
            img_modified = add_gaussian_noise(img, level)
            datatrain_2[i] = img_modified
            i += 1
        return np.append(datatrain_1, datatrain_2).reshape(dataset.shape)

    @staticmethod
    def rebuild_all_imgs(imgs, img_height, img_width):
        num_of_imgs = img_height/imgs[0].shape[0] * img_width/imgs[0].shape[1]
        list_img = []
        start_img = 0
        end_img = num_of_imgs
        while end_img <= imgs.shape[0]:
            list_img.append(rebuild_images(imgs[start_img:end_img], img_height, img_width))
            start_img += num_of_imgs
            end_img += num_of_imgs
        return np.asarray(list_img)


def downsampled(img, pixel_shape):
    if img.shape[0] % pixel_shape[0] != 0 or img.shape[1] % pixel_shape[1] != 0:
        raise ValueError('the img height and width must be multiple of the pixel height and width respectively!')
    new_height = img.shape[0] / pixel_shape[0]
    new_width = img.shape[1] / pixel_shape[1]
    down_sampled_img = np.empty((new_height, new_width))
    (h, w) = (0, 0)
    (start_x, start_y) = (0, 0)
    while start_x < img.shape[0]:
        w = 0
        start_y = 0
        while start_y < img.shape[1]:
            pixel = cut_sub_img(img, (start_x, start_y), pixel_shape[0], pixel_shape[1])
            pixel_value = int(mean_sub_img(pixel))
            down_sampled_img[h, w] = pixel_value
            w += 1
            start_y += pixel_shape[1]
        h += 1
        start_x += pixel_shape[0]
    return down_sampled_img


def rebuild_images(imgs, img_height, img_width):
    if len(imgs.shape) == 4:
        new_img = np.zeros((img_height, img_width, imgs.shape[3]))
    else:
        new_img = np.zeros((img_height, img_width))
    h, w = 0, 0
    for img in imgs:
        if h <= img_height-img.shape[0]:
            if w < img_width-img.shape[1]:
                paste_img(new_img, img, h, w)
                w += img.shape[1]
            elif w == img_width-img.shape[1]:
                paste_img(new_img, img, h, w)
                w = 0
                h += img.shape[0]
    return new_img


def paste_img(img, img_to_paste, h, w):
    for i in range(img_to_paste.shape[0]):
        for j in range(img_to_paste.shape[1]):
            if len(img.shape) == 3 and img.shape[2] == 3:
                img[h + i, w + j, 0] = img_to_paste[i, j, 0]
                img[h + i, w + j, 1] = img_to_paste[i, j, 1]
                img[h + i, w + j, 2] = img_to_paste[i, j, 2]
            else:
                img[h+i, w+j] = img_to_paste[i, j]


def process_image(dataset, img_dim, img_height, img_width, format):
    dataset = dataset.astype('float32') / 255.
    if format == 'channels_first':
        dataset = np.reshape(dataset, (len(dataset), img_dim, img_height, img_width))
    else:
        dataset = np.reshape(dataset, (len(dataset), img_height, img_width, img_dim))
    return dataset


def add_stain(img):
    noisy_image = np.copy(img)
    color = ['white', 'black', 'mean']
    stain_color = np.random.randint(3)
    stain_mask = create_stain(img)
    apply_stain(noisy_image, stain_mask, color[stain_color])
    return noisy_image


def add_gaussian_noise(img, level):
    mean = 0
    sigma = level if level != 0 else np.random.randint(0, 55)
    gauss = np.random.normal(mean, sigma, img.shape)
    boundary = np.full(img.shape, 255)
    return (img + gauss) % boundary


def apply_stain(image, mask, color):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            noise_white = np.random.randint(0, 70)
            noise_mean = np.random.randint(0, 30)
            noise_black = np.random.randint(0, 30)
            add_noise = 255 - noise_white if color == "white" else noise_black
            add_noise = 0 if mask[i, j] == 0 else add_noise
            mean_value = 100 + noise_mean - 15
            print_pixel(image, mask, i, j, i, j, add_noise, color, mean_value)


def print_pixel(img, mask, im_h, im_w, mask_h, mask_w, add_noise, color, mean_value):
    if mask[mask_h, mask_w] != 0 and color == "mean":
        img[im_h, im_w] = mean_value
    elif mask[mask_h, mask_w] != 0:
        img[im_h, im_w] = add_noise


def create_stain(img):
    if img.shape[0] < 35 or img.shape[1] < 35:
        raise ValueError('The image must have a height and a width of at least 35.')
    shape = img.shape
    mask = np.zeros(shape)
    c1 = np.random.randint(35, img.shape[0]-35)
    c2 = np.random.randint(35, img.shape[1]-35)
    a = np.random.randint(1, 30)
    b = np.random.randint(1, 30)
    for pixel1 in range(img.shape[0]):
        for pixel2 in range(img.shape[1]):
            if ellipse(pixel1, pixel2, c1, c2, a, b):
                mask[pixel1, pixel2] = 1
    return mask


def ellipse(x, y, c1, c2, a, b):
    if math.pow(x-c1,2)/math.pow(a, 2) + math.pow(y-c2, 2)/math.pow(b, 2) <= 1:
        return True
    return False


def add_scratch(image, force_color="random"):
    noisy_image = np.copy(image)
    start_point_x = np.random.randint(image.shape[1]/6, image.shape[1] - image.shape[1]/6 + 1)
    start_point_y = np.random.randint(image.shape[0]/6, image.shape[0] - image.shape[0]/6 + 1)
    length_scratch_x = np.random.randint(start_point_x, image.shape[1])
    length_scratch_y = np.random.randint(start_point_y, image.shape[0])
    scratch_shape = ['sqrt', '2x', 'sin', 'line']
    scratch_direction = ['right', 'left', 'down', 'up']
    scratch_color = ['white', 'black', 'mean']
    shape = np.random.randint(4)
    direction = np.random.randint(4)
    color = np.random.randint(3)
    if force_color == "white":
        color = 0
    elif force_color == "black":
        color = 1
    elif force_color == "mean":
        color = 2
    list_point = []
    if scratch_shape[shape] == 'sqrt':
        list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _sqrt, image.shape)
    elif scratch_shape[shape] == '2x':
        list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _2x, image.shape)
    elif scratch_shape[shape] == 'sin':
        list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _sin, image.shape)
    elif scratch_shape[shape] == 'line':
        list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _line, image.shape)
    # print(scratch_shape[shape])
    return apply_scratch(noisy_image, list_point, scratch_color[color])


def cut_sub_img(img, start_point, height, width):
    (point_h, point_w) = start_point
    if point_h + height > img.shape[0] or point_w + width > img.shape[1]:
        raise ValueError('This cut is impossible from this start_point.')
    (i, j) = (0, 0)
    sub_img = np.empty((height, width, img.shape[2])) if len(img.shape) == 3 else np.empty((height, width, 1))
    for h in range(point_h, point_h + height):
        j = 0
        for w in range(point_w, point_w + width):
            if len(img.shape) == 3 and img.shape[2] == 3:
                sub_img[i, j, 0] = img[h, w, 0]
                sub_img[i, j, 1] = img[h, w, 1]
                sub_img[i, j, 2] = img[h, w, 2]
            else:
                sub_img[i, j] = img[h, w]
            j += 1
        i += 1
    return sub_img


def mean_sub_img(img):
    mean = 0
    for pixel1 in range(img.shape[0]):
        mean += sum(img[pixel1, :])
    return float(mean)/(img.shape[0]*img.shape[1])


def func_x(start_x, length_x, start_y, length_y, direction, func, shape):
    set_point = list()
    if direction == 'up':
        for x in np.arange(0.0, length_x, 0.1):
            point_x, point_y = (start_x + int(x), start_y + int(func(x)))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'down':
        for x in np.arange(0.0, -1*length_x, -0.1):
            point_x, point_y = (start_x + int(x), start_y + int(func(x)))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'right':
        for y in np.arange(0.0, length_y, 0.1):
            point_x, point_y = (start_x + int(func(y)), start_y + int(y))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'left':
        for y in np.arange(0.0, -1*length_y, -0.1):
            point_x, point_y = (start_x + int(func(y)), start_y + int(y))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    if not set_point:
        pt1 = (0, np.random.randint(shape[1]))
        pt2 = (shape[0], np.random.randint(shape[1]))
        set_point.append(pt1)
        set_point.append(pt2)
    # print(direction, set_point)
    return set_point


def _2x(x):
    return 2*x


def _sqrt(x):
    return math.sqrt(x+4) if x >= 0 else math.sqrt(-1*x+4)


def _sin(x):
    return 16*math.sin(math.radians(x*4))


def _line(x):
    a = np.random.rand(1)[0]
    b = np.random.randint(6)-3
    return a*x + b


def apply_scratch(image, list_point, color):
    max_x = image.shape[1]
    max_y = image.shape[0]
    for x,y in list_point:
        new_value = 0
        if color == "mean":
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    point_x, point_y = (x + dx, y + dy)
                    if 0 <= point_x < 32 and 0 <= point_y < 32:
                        new_value += image[point_y, point_x]
            new_value = float(new_value) / 9
        elif color == "white":
            new_value = 255
        elif color == "black":
            new_value = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                point_x, point_y = (x + dx, y + dy)
                if 0 <= point_x < max_x and 0 <= point_y < max_y:
                    noise_white = np.random.randint(0, 70)
                    noise_mean = np.random.randint(0, 10)
                    noise_black = np.random.randint(0, 30)
                    value_apply = new_value
                    if new_value == 255:
                        value_apply = new_value - noise_white
                    elif new_value == 0:
                        value_apply = new_value + noise_black
                    elif 0 <= new_value + noise_mean - 5 < 256:
                        value_apply = new_value + noise_mean - 5
                    image[point_y, point_x] = value_apply
    return image
