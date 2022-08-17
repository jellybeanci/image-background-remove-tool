import os, torch

import PIL
from PIL import Image
from PIL import ImageFilter
from carvekit.api.high import HiInterface
from crop_image.crop import crop_object_and_strech, find_crop_edges

# interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
#                         device='cuda' if torch.cuda.is_available() else 'cpu',
#                         seg_mask_size=320, matting_mask_size=1024)

# img_path = "./leaf/leaf00009.jpg"

img_path = "./my_leaf.jpg"
# images_without_background = interface([img_path])
# leaf_wo_bg = images_without_background[0]  # og

leaf_wo_bg = Image.open("leaf_bg_rm.png")
crop_object_and_strech(leaf_wo_bg).convert("RGBA").resize((256, 256)).save("leaf_bg_rm_rgb_crp_strech.png")

# cropped = crop_object_and_strech(leaf_wo_bg)
# cropped.save('leaf_op.png')

# filtered_img.save('leaf_gauss.png')
# leaf_wo_bg.save('leaf_og.png')  # og
