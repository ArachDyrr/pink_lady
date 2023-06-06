import torch
from glob import glob
import shutil
import os

img_path = r"C:\Users\jiyoo\workspace\MakeAIWork2\photo_image\original_image"

img_list = glob(img_path + '\*.jpg')
img_list.extend(glob(img_path + '\*.png'))

print(img_list)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img_move_path = r"C:\Users\jiyoo\workspace\MakeAIWork2\photo_image\only_ people_image"

for img_path in img_list:
    results = model(img_path)
    print(img_path)
    for pred in results.pred[0]:
        tag = results.names[int(pred[-1])] 
        print(tag)
        if tag == "person":
            print("move")
            shutil.move(img_path, img_move_path + '\\' + os.path.basename(img_path))
            break