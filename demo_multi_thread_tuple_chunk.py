import os, time, torch
from multiprocessing.pool import ThreadPool
from carvekit.api.high import HiInterface

srcPath = "leaf"
DEST_PATH = "results"

paths = os.listdir(srcPath)  # returns list
full_paths = list(map(lambda path: os.path.join("./", srcPath, path), paths))
filter_path = [i for i in full_paths if i.endswith('.jpg') or i.endswith('.png') or i.endswith('jpeg')]

# filter_path = filter_path[0: 32]  # select first 10 item
tuple_list = list(enumerate(filter_path))

interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=320, matting_mask_size=1024)

os.makedirs(DEST_PATH, exist_ok=True)


def image_process(img_index_and_path_arr):
    start_t = time.perf_counter()
    print(f"\n'now working on images between {img_index_and_path_arr[0][0]} and {img_index_and_path_arr[-1][0]}'")
    for img_index_and_path in img_index_and_path_arr:
        index, img_path = img_index_and_path
        image = interface([img_path])
        f_name = f"img_{index}.png"
        dest = os.path.join(DEST_PATH, f_name)
        image.save(dest)
        end_t = time.perf_counter()
        return f_name, end_t - start_t


def get_sublists(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def split_list(lst, n):
    return list(get_sublists(lst, n))


# start_g_t = time.perf_counter()
# with ThreadPool() as pool:
#     results = pool.imap_unordered(image_process, tuple_list)
#     for f_name, duration in results:
#         print(f"{f_name} completed in {duration:.2f} seconds")
# end_g_t = time.perf_counter()
# g_duration = end_g_t - start_g_t
# print(f"process done in {g_duration:.2f} seconds")


