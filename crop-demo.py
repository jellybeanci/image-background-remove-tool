import os
from PIL import Image
from crop_image.crop import crop_object_and_strech


def is_windows():
  return os.name == 'nt'


def get_name_path(img_path: str) -> str:
  return img_path.split("\\" if is_windows() else "/")[-1].split(".")[0]


src = "cat.png"
img_name = get_name_path(src)
f_name = f"{img_name}_crp.png"
imread: Image = Image.open(src)
og_size = imread.size
crop_object_and_strech(imread).resize(og_size).save(f_name)
