import os, re
from typing import List

from PIL import Image
import numpy as np

# srcPath = r"tomato/combined_train"
srcPath = r"demo_t"
DEST_PATH = "results"


def get_path_name(path: str) -> str:
    return path.split('/')[-1].split('.')[0]


def get_class_name(img_name: str) -> str:
    return re.search(r"(^[A-Za-z_-]+)(\d+)", img_name).groups()[0]


def read_dir_as_list(src: str):
    paths = os.listdir(src)  # returns list
    full_paths = list(map(lambda path: os.path.join("./", src, path), paths))
    filter_ext = [i for i in full_paths if i.endswith('.jpg') or i.endswith('.png') or i.endswith('jpeg')]
    return filter_ext


def read_image_as_array(img_path: str):
    img = Image.open(img_path)
    im_arr = np.asarray(img)
    im_class = get_class_name(get_path_name(img_path))
    return im_arr, im_class


def read_dir_as_image_array(img_path_list: List[str]):
    imgs = []
    im_classes = []
    for img_path in img_path_list:
        im_arr, im_class = read_image_as_array(img_path)
        imgs.append(im_arr)
        im_classes.append(im_class)
    return imgs, im_classes


filter_path = read_dir_as_list(srcPath)
img_names = list(map(lambda path: get_path_name(path), filter_path))
class_names = list(map(lambda img_name: get_class_name(img_name), img_names))

# print(filter_path)
# print(img_names)
# print(class_names)

im_list, class_list = read_dir_as_image_array(read_dir_as_list(srcPath))

npyimage = np.array(im_list)
np.savez_compressed("tomato", npyimage)
npylabel = np.array(class_list)
np.savez_compressed("label", npylabel)