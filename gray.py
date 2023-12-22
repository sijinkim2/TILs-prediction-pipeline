import os
#os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
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
    #if val_img(img == 255) > 0.6:
    #    img_array = np.array(img)
    #plt.hist(img_array)
    #plt.show()
    #cv2.imwrite('/home/skim/NAS01/Databank/public/2_Pathology/whole_body/cropped_WSI/TCGA-BRCA/TCGA-A2-A0YK-01A-02-BSB/gray_scale_thre/%s' % (dst), img)
    #img.save(("gray/%s" % (dst)))



'''
for i in files:
    path_img = os.path.join('/home/skim/NAS01/Databank/2_Pathology/public/TIGER/wsitils/images', i)
    file_name = Path(path_img).stem
    path_mask = os.path.join(
        '/home/skim/NAS01/Databank/2_Pathology/public/TIGER/wsitils/tissue-masks',
        i)
    #img = Image.open(path_img)
    #img_mask = Image.open(path_mask)
    img = cv2.imread
    img_mask = cv2.imread(path_mask)
    maskarray = np.array(img_mask)
    #img = cv2.imread(path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = 1
    h, w, c = img_mask.shape
    #height
    height = round(1 + (h / 256))
    difference_h = h / height
    overlap_h = (256 - difference_h) / (height - 1)

    #width
    width = round(1 + (w / 256))
    difference_w = w / width
    overlap_w = (256 - difference_w)/(width - 1)


    left = 0
    up = 0
    right = 256
    down = 256

    img = Image.fromarray(np.array(img, dtype='uint8'))
    img_mask = Image.fromarray(np.array(img_mask, dtype='uint8'))

    for j in range(height):
        for k in range(width):
            dst = str(file_name)  + '_' + str(a) + '.png'
            croppedImage = img.crop((left, up, right, down))
            croppedImage_mask = img_mask.crop((left, up, right, down))
            #croppedImage.show()
            #img_array = np.array(img)
            croppedImage_array = np.array(croppedImage_mask)
            #print(" %d, %d,잘려진 사진 크기 :" %(k, a),croppedImage.size)

            pixel_amount = np.count_nonzero(croppedImage_array == 255)

            if (pixel_amount > ((256*256)*0.65)):
                croppedImage.save(("/home/skim/NAS01/Users/skim/Tiger_crop/WSITILS/%s" % (dst)))
            else:
                print('%s는 백그라운드 비율이 더 크므로 저장 안됨'%(dst))
            a = a + 1
            left = left + (difference_w - overlap_w)
            right = right + (difference_w - overlap_w)
        up = up + (difference_h - overlap_h)
        down = down + (difference_h - overlap_h)
        left = 0
        right = 256
'''
