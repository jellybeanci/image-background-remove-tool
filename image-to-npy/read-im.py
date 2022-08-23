from PIL import Image
import random
import numpy as np

tomatoes = np.load(r"tomato.npz")['arr_0']
labels = np.load(r"label.npz")['arr_0']


rnd = random.randint(0, 46)
rnd_arr = tomatoes[rnd]
print(rnd_arr.size)
print(rnd_arr.shape)
print(rnd_arr.dtype)
img = Image.fromarray(rnd_arr)
img.show()
print(labels[rnd])

# for label in labels:
#     print(label)
