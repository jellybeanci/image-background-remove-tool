import os

from PIL import Image
import numpy as np


def get_dest(img_name, default = r"comp2"):
    return os.path.join(default, img_name)

# Read Image from Disk
img_1 = Image.open(get_dest("OG_1.jpg"))
img_2 = Image.open(get_dest("OG_2.jpg"))

# Save as JPEG
img_1.save(get_dest("JPEG_1.jpg"))
img_2.save(get_dest("JPEG_2.jpg"))

# Save as PNG
img_1.save(get_dest("PNG_1.png"))
img_2.save(get_dest("PNG_2.png"))

# Covnert to np array
np_img_1 = np.asarray(img_1)
np_img_2 = np.asarray(img_2)

# Save as NPY
np.save(get_dest("NPY_1.npy"), np_img_1)
np.save(get_dest("NPY_2.npy"), np_img_2)

# Save as NPZ compresses
np.savez_compressed(get_dest("NPZ_1"), np_img_1)
np.savez_compressed(get_dest("NPZ_2"), np_img_2)

# Create Image list for multiple array
img_list = [np_img_1, np_img_2]
img_list_np = np.asarray(img_list)

#
np.save(get_dest("NP_LIST_NPY.npy"), img_list_np)

np.savez_compressed(get_dest("NP_LIST_NPZ"), img_list_np)