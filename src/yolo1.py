import torch
from glob import glob
import sys


img_path = "apple_disease_classification"

img_list = glob(img_path + '/*.jpg')
img_list.extend(glob(img_path + '/*.png'))

print(img_list)

sys.path.insert(0, 'C:/Users/jiyoo/workspace/MakeAIWork3')


from models.experimental import attempt_load
model = attempt_load('yolov5s', map_location=torch.device('cpu'))


for img_path in img_list:
    results = model(img_path)
    print(img_path)
    
    # Apple herkennen met YOLOv5
    apple_results = results.pred[0][results.pred[0][:, -1] == 0]  # Filter resultaten voor appel (klasse 0)

    # Masker toepassen op de appelvorm
    masks = apple_results[:, 4:].argmax(dim=1).unsqueeze(1).float()
    masked_results = torch.cat((apple_results[:, :4], masks), dim=1)

    # Uitvoer opslaan
    masked_results.save("image_control", img_path)

    for pred in apple_results:
        tag = results.names[int(pred[-1])]
        print(tag)
