import os, re
from PIL import Image
import numpy as np

srcPath = "img_demo"
DEST_PATH = "results"


def get_path_name(path: str) -> str:
    return path.split('/')[-1].split('.')[0]


def get_class_name(img_name: str) -> str:
    return re.search(r"(^[A-Za-z_-]+)(\d+)", img_name).groups()[0]


paths = os.listdir(srcPath)  # returns list
full_paths = list(map(lambda path: os.path.join("./", srcPath, path), paths))
filter_path = [i for i in full_paths if i.endswith('.jpg') or i.endswith('.png') or i.endswith('jpeg')]
img_names = list(map(lambda path: get_path_name(path), filter_path))
class_names = list(map(lambda img_name: get_class_name(img_name), img_names))

print(filter_path)
print(img_names)
print(class_names)

imgs = []
for img_path in filter_path:
    img = Image.open(img_path)
    im_arr = np.asarray(img)
    imgs.append(im_arr)

npyimage = np.array(imgs)

np.save("tomato.npy", npyimage)