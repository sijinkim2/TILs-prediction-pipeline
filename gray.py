import os
from PIL import Image
import numpy as np
from pathlib import Path
import collections
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

files = os.listdir(
    "image_path")
a = 1


# Remove Background

for i in files:
    path = os.path.join('image_path', i)
    #img = Image.open(path).convert('L')
    name = Path(path).stem
    dst = str(name) + '.png'
    img = cv2.imread(path)

    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = str(i)
    img_array = np.array(img2)

    img2 = ((img_array < 210) * 255).astype(np.uint8)
    print(a)
    pixel_amount = np.count_nonzero(img2 == 255)
    if (pixel_amount > ((256 * 256) * 0.5)): 
        cv2.imwrite(
            'save_path' % (
                dst), img)

    a = a + 1



