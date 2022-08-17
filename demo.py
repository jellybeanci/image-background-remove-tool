import os, torch
from carvekit.api.high import HiInterface

interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=320, matting_mask_size=1024)

img_path = "./leaf/leaf00000.jpg"
images_without_background = interface([img_path])
leaf_wo_bg = images_without_background[0]
leaf_wo_bg.save('leaf.png')