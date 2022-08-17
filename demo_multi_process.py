import os
import time
from multiprocessing import Pool

import torch
from carvekit.api.high import HiInterface

import secrets

srcPath = "leaf"
DEST_PATH = "results"

paths = os.listdir(srcPath)  # returns list
full_paths = list(map(lambda path: os.path.join("./", srcPath, path), paths))
filter_path = [i for i in full_paths if i.endswith('.jpg')]

interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=320, matting_mask_size=1024)

os.makedirs(DEST_PATH, exist_ok=True)

def image_process(img_path):
    start_t = time.perf_counter()
    print(f"'now working on image at {img_path}'")
    image = interface([img_path])[0]
    f_name = f"img_{secrets.token_urlsafe(5)}.png"
    dest = os.path.join(DEST_PATH, f_name)
    image.save(dest)
    end_t = time.perf_counter()
    return f_name, end_t - start_t

def job():
    with Pool() as pool:
        results = pool.imap_unordered(image_process, filter_path)

        for f_name, duration in results:
            print(f"{f_name} completed in {duration:.2f} seconds")

def main():
    job()

if __name__ == '__main__':
    main()