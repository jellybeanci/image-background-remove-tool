import os
from multiprocessing.pool import ThreadPool
from PIL import Image

from crop_image.crop import crop_object_and_strech

srcPath = "results"
DEST_PATH = "results_crp"

paths = os.listdir(srcPath)  # returns list
full_paths = list(map(lambda path: os.path.join("./", srcPath, path), paths))
filter_path = [i for i in full_paths if i.endswith('.jpg') or i.endswith('.png') or i.endswith('jpeg')]
filter_path = filter_path[0:20]
tuple_list = list(enumerate(filter_path))

os.makedirs(DEST_PATH, exist_ok=True)

def get_file_name(img_path):
    return img_path.split('/')[-1].split('.')[0]


def image_crop_process(img_index_and_path):
    index, img_path = img_index_and_path
    img_name = get_file_name(img_path)
    f_name = f"{img_name}_crp.png"
    dest = os.path.join(DEST_PATH, f_name)
    imread: Image = Image.open(img_path)
    og_size = imread.size
    return crop_object_and_strech(imread).resize(og_size), dest

with ThreadPool() as pool:
    results = pool.imap_unordered(image_crop_process, tuple_list)

    for crp_img, dest in results:
        crp_img.save(dest)