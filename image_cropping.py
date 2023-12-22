import os
from PIL import Image
import numpy as np
from pathlib import Path
import collections

files = os.listdir(
    "/home/skim/Rakib/masks")



for i in files:
    path = os.path.join('image_path', i)
    file_name = Path(path).stem
    img = Image.open(path)#.convert("RGB")


    a = 1
    w, h = img.size

    # width
    width = round(1 + (w / 256))
    #width = round(w / 256)
    difference_w = w / width
    overlap_w = (256 - difference_w) / (width - 1)

    #height
    height = round(1 + (h / 256))
    #height = round(h / 256)
    difference_h = h / height
    overlap_h = (256 - difference_h) / (height - 1)



    left = 0
    up = 0
    right = 256
    down = 256


    for j in range(height):
        for k in range(width):
            dst = str(file_name)  + '_' + str(a) + '.png'
            croppedImage = img.crop((left, up, right, down))
            #croppedImage.show()
            #img_array = np.array(img)
            croppedImage_array = np.array(croppedImage)
            print(" %d, %d,잘려진 사진 크기 :" %(k, a),croppedImage.size)
            croppedImage.save(("image_path" % (dst)))
            a = a + 1
            left = left + (difference_w - overlap_w)
            right = right + (difference_w - overlap_w)
        up = up + (difference_h - overlap_h)
        down = down + (difference_h - overlap_h)
        left = 0
        right = 256


