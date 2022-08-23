import os, time, torch
from multiprocessing.pool import ThreadPool
from carvekit.api.high import HiInterface

srcPath = "leaf"
DEST_PATH = "results"

paths = os.listdir(srcPath)  # returns list
full_paths = list(map(lambda path: os.path.join("./", srcPath, path), paths))
filter_path = [i for i in full_paths if i.endswith('.jpg') or i.endswith('.png') or i.endswith('jpeg')]

# filter_path = filter_path[0: 10]  # select first 10 item

os.makedirs(DEST_PATH, exist_ok=True)

interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=320, matting_mask_size=1024)


def is_windows():
    return os.name == 'nt'


def get_name_path(img_path: str) -> str:
    return img_path.split("\\" if is_windows() else "/")[-1].split(".")[0]


def image_process(img_path):
    start_t = time.perf_counter()
    img_name = get_name_path(img_path)
    print(f"\n'now working on image at {img_name}'")
    image = interface([img_path])[0]
    f_name = f"{img_name}_crp.png"
    dest = os.path.join(DEST_PATH, f_name)
    image.save(dest)
    end_t = time.perf_counter()
    return f_name, end_t - start_t


start_g_t = time.perf_counter()
with ThreadPool() as pool:
    results = pool.imap_unordered(image_process, filter_path)
    for f_name, duration in results:
        print(f"{f_name} completed in {duration:.2f} seconds")
end_g_t = time.perf_counter()
g_duration = end_g_t - start_g_t
print(f"process done in {g_duration:.2f} seconds")
